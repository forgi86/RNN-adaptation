import os
from typing import List, Union
import numpy as np
import time
import sys
from pathlib import Path
import pickle
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
import torch
from torch import nn
import matplotlib.pyplot as plt

from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from diffutil.jacobian import parameter_jacobian
import examples.RLC.loader as loader
from diffutil.products import jvp, unflatten_like
from examples.RLC.utils import get_time_str, StateSpaceWrapper, setup_run_dir
from torchid import metrics

BASE_DIR = '/system/user/publicdata/meta_learning/ode/rlc_dataset-resistor_range1_14-capacitor_range100_800-inductor_range20_140'

class JFRTester(object):

    def __init__(self, experiment_name: str, base_dataset_dir: str = BASE_DIR, seed: int = 0):
        np.random.seed(0)
        torch.manual_seed(0)
        self._run_dir = setup_run_dir(experiment_name)
        self._base_dataset_dir = base_dataset_dir
        pretrain_model = "/system/user/beck/pwbeck/meta/rnn-adaptation-multitask/models/ss_model.pt"
        self._pretrain_model_path = pretrain_model
        self._noise_std_output = 0.1

        self._input_size = 1
        self._output_size = 1
        self._model = self._get_model()

    def _get_model(self,
                   n_x: int = 2,
                   n_u: int = 1,
                   n_feat: int = 50) -> nn.Module:

        assert self._pretrain_model_path.exists(
        ), "The specified pretrain model does not exist. Consider pretraining first."
        ss_model = NeuralStateSpaceModel(n_x=n_x, n_u=n_u, n_feat=n_feat)
        nn_solution = ForwardEulerSimulator(ss_model)
        nn_solution.ss_model.load_state_dict(
            torch.load(str(self._pretrain_model_path)))

        model_wrapped = StateSpaceWrapper(nn_solution)
        return model_wrapped

    def evaluate(self,
                 split: str,
                 support_sizes: List[int],
                 n_datasets: int = None):
        datasets_dir = Path(self._base_dataset_dir) / split
        assert datasets_dir.exists()

        datasets = list(datasets_dir.glob('*.npy'))
        if n_datasets:
            datasets = datasets[:n_datasets]
        assert len(datasets) > 0
        print(len(datasets))

        eval_results = {}

        for dataset in datasets:
            ds_filename = split + '/' + dataset.name
            eval_results[ds_filename] = {}
            pbar = tqdm(support_sizes)
            pbar.set_description(f'Dataset {ds_filename}')
            for support_size in pbar:
                theta_lin = self._calculate_linear_adaption(
                    ds_filename=ds_filename, support_size=support_size)
                results = self._evaluate_linear_adaption(
                    theta_lin, ds_filename)
                eval_results[ds_filename][support_size] = results

        # save eval results
        with open(self._run_dir/ 'eval_results.p', 'wb') as f:
            pickle.dump(eval_results, f)

        return eval_results

    def pretrain(self):
        pass

    def _calculate_linear_adaption(self,
                                   ds_filename: str,
                                   support_size: int,
                                   vectorize: bool = True) -> torch.Tensor:
        # In[Load dataset]
        t, u, y, x = loader.rlc_loader_multitask(
            ds_filename,
            trajectory=0,
            steps=support_size,
            noise_std=self._noise_std_output,
            scale=False, 
            base_dir=self._base_dataset_dir)

        seq_len = t.size
        time_start = time.time()

        # In[Setup neural model structure and load fitted model parameters]

        u_torch = torch.tensor(u[None, ...],
                               dtype=torch.float,
                               requires_grad=False)
        y_torch = torch.tensor(y[None, ...], dtype=torch.float)
        u_torch_f = torch.clone(u_torch.view(
            (1 * seq_len,
             self._input_size)))  # [bsize*seq_len, n_in] # [2000,1]
        y_torch_f = torch.clone(y_torch.view(
            1 * seq_len, self._output_size))  # [bsize*seq_len, ]
        # In[Adaptation in parameter space (naive way)]
        J = parameter_jacobian(
            self._model, u_torch_f,
            vectorize=vectorize).detach().numpy()  # full parameter jacobian
        n_param = J.shape[1]
        Ip = np.eye(n_param)
        F = J.transpose() @ J
        A = F + self._noise_std_output**2 * Ip
        theta_lin = np.linalg.solve(A, J.transpose() @ y)  # adaptation!

        adapt_time = time.time() - time_start
        # print(f"\nAdapt time: {adapt_time:.2f}")

        return torch.from_numpy(theta_lin)

    def _evaluate_linear_adaption(self,
                                  theta_lin: torch.Tensor,
                                  ds_filename: str,
                                  query_size: int = 2000):
        t_new, u_new, y_new, x_new = loader.rlc_loader_multitask(
            ds_filename,
            trajectory=1,
            steps=query_size,
            noise_std=self._noise_std_output,
            scale=False, 
            base_dir=self._base_dataset_dir)

        seq_len_new = t_new.size

        # In[Model wrapping]
        u_torch_new = torch.tensor(u_new[None, :, :])
        u_torch_new_f = torch.clone(
            u_torch_new.view(
                (1 * seq_len_new, self._input_size)))  # [bsize*seq_len, n_in]

        # In[Nominal model output]
        y_sim_new_f = self._model(u_torch_new_f)
        y_sim_new = y_sim_new_f.reshape(seq_len_new,
                                        self._output_size).detach().numpy()

        # In[Parameter jacobian-vector product]
        theta_lin_f = unflatten_like(theta_lin,
                                     tensor_lst=list(self._model.parameters()))
        y_lin_new_f = jvp(y_sim_new_f, self._model.parameters(),
                          theta_lin_f)[0]
        y_lin_new = y_lin_new_f.reshape(seq_len_new,
                                        self._output_size).detach().numpy()

        # In[Plot]
        plt.plot(y_new, 'k', label="True")
        plt.plot(y_sim_new, 'r', label="Sim")
        plt.plot(y_lin_new, 'b', label="Lin-Sense")
        plt.legend()
        plt.grid()
        plt.xlim(0, 500)
        # plt.show()
        datetime_str = get_time_str()
        # plt.savefig(f"fig/rlc_eval_results_{datetime_str}.pdf",
        #             bbox_inches='tight')

        # Saving state and input
        save_path = Path.cwd() / 'data' / 'RLC_SS_NL'
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path / "04_eval_y.npy"), y_new)
        np.save(str(save_path / "04_eval_y_sim.npy"), y_sim_new)
        np.save(str(save_path / "04_eval_y_lin.npy"), y_lin_new)

        # R-squared and MSE metrics
        r_sq_lin = metrics.r_squared(y_new, y_lin_new)
        # print(f"R-squared linear model: {r_sq_lin}")

        r_sq_nom = metrics.r_squared(y_new, y_sim_new)
        # print(f"R-squared nominal model: {r_sq_nom}")

        mse_lin = np.mean((y_new - y_lin_new)**2, axis=0)
        # print(f"MSE linear model: {mse_lin}")
        mse_nom = np.mean((y_new - y_sim_new)**2, axis=0)
        # print(f"MSE nominal model: {mse_nom}")

        # results = {
        #     ('adapted_model', 'rsquared'): r_sq_lin.item(),
        #     ('adapted_model', 'mse'): mse_lin.item(),
        #     ('pretrained_model', 'rsquared'): r_sq_nom.item(),
        #     ('pretrained_model', 'mse'): mse_nom.item()
        # }
        results_mse = {'jfr_model': mse_lin.item(),
            'no_finetune': mse_nom.item()
        }
        results_rsquared = {'jfr_model': r_sq_lin.item(),
            'no_finetune': r_sq_nom.item()
        }
        results = {'mse': results_mse, 'rsquared': results_rsquared}
        return results


if __name__ == '__main__':

    model_path = "models/ss_model.pt"
    BASE_DIR = '/system/user/publicdata/meta_learning/ode/rlc_dataset-resistor_range1_14-capacitor_range100_800-inductor_range20_140'
    datasets_dir = Path(BASE_DIR) / 'test'

    support_sizes = [20, 30, 100]

    jfr = JFRTester(model_path)
    # jfr.pretrain() # TODO
    jfr.evaluate(base_dataset_dir=BASE_DIR,
                 split='test',
                 support_sizes=support_sizes,
                 n_datasets=3)
