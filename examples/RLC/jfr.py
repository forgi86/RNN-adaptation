import os
from typing import List, Dict
import numpy as np
import pandas as pd
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


def eval_results_to_df(eval_results: Dict[str, Dict[int, Dict]], metric='mse'):
    results = {}
    model_type_index = None
    for ds_name, ds_res in eval_results.items():
        results[ds_name] = {(support_size, mt): ds_ss_res[metric][mt]
                            for support_size, ds_ss_res in ds_res.items()
                            for mt in ds_ss_res[metric]}

    multi_ind = pd.MultiIndex.from_tuples(results[ds_name].keys(),
                                          names=['support_size', 'model_type'])
    return pd.DataFrame(results, index=multi_ind).transpose()


class JFRTester(object):

    def __init__(self,
                 experiment_name: str,
                 base_dataset_dir: str = BASE_DIR,
                 run_dir: str = None,
                 seed: int = 0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._run_dir = setup_run_dir(experiment_name, run_dir)
        self._base_dataset_dir = base_dataset_dir
        self._noise_std_output = 0.1

        self._input_size = 1
        self._output_size = 1
        self._model = None

    def _get_model(self,
                   n_x: int = 2,
                   n_u: int = 1,
                   n_feat: int = 50,
                   pretrain_model_path: Path = None,
                   wrapped: bool = False) -> nn.Module:

        ss_model = NeuralStateSpaceModel(n_x=n_x, n_u=n_u, n_feat=n_feat)
        nn_solution = ForwardEulerSimulator(ss_model)
        if pretrain_model_path:
            assert pretrain_model_path.exists(
            ), "The specified pretrain model does not exist. Consider pretraining first."
            nn_solution.ss_model.load_state_dict(
                torch.load(str(pretrain_model_path)))

        if wrapped:
            nn_solution = StateSpaceWrapper(nn_solution)
        return nn_solution

    def evaluate(self,
                 split: str,
                 support_sizes: List[int],
                 n_datasets: int = None):
        # load pretrained model
        self._model = self._get_model(pretrain_model_path=self._run_dir /
                                      "ss_model.pt",
                                      wrapped=True)
        datasets_dir = Path(self._base_dataset_dir) / split
        assert datasets_dir.exists()

        datasets = list(datasets_dir.glob('*.npy'))
        if n_datasets:
            datasets = datasets[:n_datasets]
        assert len(datasets) > 0
        print(f'Number of datasets: {len(datasets)}')

        eval_results = {}

        for i, dataset in enumerate(datasets):
            ds_filename = split + '/' + dataset.name
            eval_results[ds_filename] = {}
            pbar = tqdm(support_sizes)
            pbar.set_description(f'ds({i}): {ds_filename}')
            for support_size in pbar:
                theta_lin = self._calculate_linear_adaption(
                    ds_filename=ds_filename, support_size=support_size)
                results = self._evaluate_linear_adaption(
                    theta_lin, ds_filename, support_size=support_size)
                eval_results[ds_filename][support_size] = results

        # save eval results
        with open(self._run_dir / 'eval_results.p', 'wb') as f:
            pickle.dump(eval_results, f)

        for metric in ['mse', 'rsquared']:
            df = eval_results_to_df(eval_results, metric=metric)
            df.to_csv(self._run_dir / f'eval_results_{metric}.csv')

        return eval_results

    def pretrain(self,
                 ds_filename: str = 'train/R:3.0_L:5e-05_C:2.7e-07.npy',
                 num_timesteps: int = 2000,
                 num_trajectories: int = 3,
                 val_trajectory: int = 0):
        # Overall parameters
        model_filename = self._run_dir / "ss_model.pt"
        hidden_filename = self._run_dir / "ss_hidden.pt"
        if model_filename.exists():
            print("A pretrained model exists already. Skipping pretraining..")
            return
        num_iter = 10000  # gradient-based optimization steps
        seq_len = 256  # subsequence length m
        batch_size = 16  # batch size q
        t_fit = 2e-3  # fitting on t_fit ms of data
        alpha = 1.0  # regularization weight
        lr = 1e-4  # learning rate
        test_freq = 100  # print message every test_freq iterations
        var_idx = 0  # voltage

        t, u, y, x = loader.rlc_loader_multitask(
            ds_filename,
            trajectory=0,
            trajectory_stop=num_trajectories,
            steps=num_timesteps,
            noise_std=0.1,
            scale=False)
        # Get fit data
        u_fit = u
        y_fit = y
        time_fit = t

        # Fit data to pytorch tensors #
        u_torch_fit = torch.from_numpy(u_fit)
        time_torch_fit = torch.from_numpy(time_fit)
        x_hidden_fit = torch.tensor(np.c_[y_fit, np.zeros_like(y_fit)],
                                    requires_grad=True)

        # Setup neural model structure
        nn_solution = self._get_model(wrapped=False)

        # Setup optimizer
        params_net = list(nn_solution.ss_model.parameters())
        params_hidden = [x_hidden_fit]
        optimizer = torch.optim.Adam([
            {
                'params': params_net,
                'lr': lr
            },
            {
                'params': params_hidden,
                'lr': 10 * lr
            },
        ],
                                     lr=lr)

        # Batch extraction funtion
        def get_batch(batch_size, seq_len):

            # Select batch indexes
            num_train_samples = y_fit.shape[1]
            batch_traj = np.random.choice(np.arange(num_trajectories),
                                          batch_size,
                                          replace=True)
            batch_start = np.random.choice(
                np.arange(num_train_samples - seq_len, dtype=np.int64),
                batch_size,
                replace=False)  # batch start indices
            batch_idx = batch_start[:, np.newaxis] + np.arange(
                seq_len)  # batch samples indices
            batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

            # Extract batch data
            batch_t = torch.tensor(time_fit[batch_traj, batch_idx])
            batch_x0_hidden = x_hidden_fit[batch_traj, batch_start, :]
            batch_x_hidden = x_hidden_fit[batch_traj, batch_idx]
            batch_u = torch.tensor(u_fit[batch_traj, batch_idx])
            batch_y = torch.tensor(y_fit[batch_traj, batch_idx])

            return batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden

            # Scale loss with respect to the initial one

        with torch.no_grad():
            batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(
                batch_size, seq_len)
            batch_x_sim = nn_solution(batch_x0_hidden, batch_u)
            batch_y_sim = batch_x_sim[..., [var_idx]]
            traced_nn_solution = torch.jit.trace(nn_solution,
                                                 (batch_x0_hidden, batch_u))
            err_init = batch_y_sim - batch_y
            scale_error = torch.sqrt(torch.mean(err_init**2, dim=(0, 1)))

        LOSS = []
        LOSS_CONSISTENCY = []
        LOSS_FIT = []
        start_time = time.time()
        # Training loop

        scripted_nn_solution = torch.jit.script(nn_solution)
        pbar = tqdm(range(0, num_iter))
        for itr in pbar:

            optimizer.zero_grad()

            # Simulate
            batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(
                batch_size, seq_len)
            # batch_x_sim = traced_nn_solution(batch_x0_hidden, batch_u) # 52 seconds RK | 13 FE
            # batch_x_sim = nn_solution(batch_x0_hidden, batch_u) # 70 seconds RK | 13 FE
            batch_x_sim = scripted_nn_solution(
                batch_x0_hidden, batch_u)  # 71 seconds RK | 13 FE

            # Compute fit loss
            batch_y_sim = batch_x_sim[..., [var_idx]]
            err_fit = batch_y_sim - batch_y
            err_fit_scaled = err_fit / scale_error
            loss_fit = torch.mean(err_fit_scaled**2)  # Loss function: MSE

            # Compute consistency loss
            err_consistency = batch_x_sim - batch_x_hidden
            err_consistency_scaled = err_consistency / scale_error
            loss_consistency = torch.mean(err_consistency_scaled**2)

            # Compute trade-off loss
            loss = loss_fit + alpha * loss_consistency
            # loss = loss_fit

            # Statistics
            LOSS.append(loss.item())
            LOSS_CONSISTENCY.append(loss_consistency.item())
            LOSS_FIT.append(loss_fit.item())
            # if itr % test_freq == 0:
            pbar.set_description_str(
                f'Loss {loss.item():.4f} Cons. Loss {loss_consistency.item():.4f} Fit Loss {loss_fit.item():.4f}'
            )

            # Optimize
            loss.backward()
            optimizer.step()

        train_time = time.time() - start_time
        print(f"\nTrain time: {train_time:.2f}")

        #* save model
        torch.save(nn_solution.ss_model.state_dict(), model_filename)
        torch.save(x_hidden_fit, hidden_filename)

        #* eval pretraining
        input_data_val = u[val_trajectory, 0:num_timesteps]
        state_data_val = x[val_trajectory, 0:num_timesteps]

        x0_val = np.zeros(2, dtype=np.float32)
        x0_torch_val = torch.from_numpy(x0_val)
        u_torch_val = torch.tensor(input_data_val)
        x_true_torch_val = torch.from_numpy(state_data_val)

        with torch.no_grad():
            x_sim_torch_val = nn_solution(x0_torch_val[None, :],
                                          u_torch_val[:, None, :])
            x_sim_torch_val = x_sim_torch_val.squeeze(1)

        #* plot pretraining results
        plot_dir = self._run_dir / 'plots_pretrain'
        plot_dir.mkdir(exist_ok=True, parents=True)
        # plot predictions
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(np.array(x_true_torch_val[:, 0]), label='True')
        ax[0].plot(np.array(x_sim_torch_val[:, 0]), label='Fit')
        ax[0].legend()
        ax[0].grid(True)
        ax[0].set_ylabel("output voltage v_c")

        ax[1].plot(np.array(x_true_torch_val[:, 1]), label='True')
        ax[1].plot(np.array(x_sim_torch_val[:, 1]), label='Fit')
        ax[1].legend()
        ax[1].grid(True)
        ax[1].set_ylabel("current i_l")

        ax[2].plot(np.array(u_torch_val), label='Input')
        ax[2].grid(True)
        ax[2].set_ylabel("input v_in")
        fig.savefig(str(plot_dir / 'rlc_state_space_predictions.png'),
                    bbox_inches='tight',
                    dpi=300)

        # plot losses
        fig, ax = plt.subplots(1, 1)
        ax.plot(LOSS, 'k', label='ALL')
        ax.plot(LOSS_CONSISTENCY, 'r', label='CONSISTENCY')
        ax.plot(LOSS_FIT, 'b', label='FIT')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("Loss (-)")
        ax.set_xlabel("Iteration (-)")

        fig.savefig(str(plot_dir /
                        f"rlc_pretrain_loss_batchseqlen{seq_len}.png"),
                    bbox_inches='tight',
                    dpi=300)

        # plot hidden state
        x_hidden_fit_np = x_hidden_fit.detach().numpy()
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(x[val_trajectory, :, 0], 'k', label='True')
        # ax[0].plot(x_fit[:, 0], 'b', label='Measured')
        ax[0].plot(x_hidden_fit_np[val_trajectory, :, 0], 'r', label='Hidden')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(x[val_trajectory, :, 1], 'k', label='True')
        # ax[1].plot(x_fit[:, 1], 'b', label='Measured')
        ax[1].plot(x_hidden_fit_np[val_trajectory, :, 1], 'r', label='Hidden')
        ax[1].legend()
        ax[1].grid(True)
        fig.savefig(str(plot_dir / "rlc_learned_hidden_state_preds.png"),
                    bbox_inches='tight',
                    dpi=300)

        #* save predictions
        preds_dir = self._run_dir / 'predictions_pretrain'
        preds_dir.mkdir(exist_ok=True, parents=True)
        np.savez(preds_dir / 'predictions_pretrain.npz',
                 ss_ground_truth=np.array(x_true_torch_val),
                 ss_predictions=np.array(x_sim_torch_val),
                 inputs_u=np.array(u_torch_val),
                 ss_learned_consistency=x_hidden_fit_np)
        # np.save(preds_dir / "state_space_ground_truth.npy",
        #         np.array(x_true_torch_val))
        # np.save(preds_dir / "state_space_predictions.npy",
        #         np.array(x_sim_torch_val))
        # np.save(preds_dir / "inputs_u.npy", np.array(u_torch_val))
        # np.save(preds_dir / "state_space_learned_consistency.npy",
        #         x_hidden_fit_np)
        plt.close()

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
                                  support_size: int,
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

        #* plot evaluation results
        plot_dir = self._run_dir / 'plots_eval'
        plot_dir.mkdir(exist_ok=True, parents=True)
        fig, ax = plt.subplots(1, 1)
        ax.plot(y_new, 'k', label="ground_truth")
        ax.plot(y_sim_new, 'r', label="no_finetune")
        ax.plot(y_lin_new, 'b', label="jfr_adapted")
        ax.legend()
        ax.grid()
        ax.set_xlim(0, 500)
        datetime_str = get_time_str()
        fig.savefig(str(plot_dir / f'plot_ds-{ds_filename.replace("/","_")}_support-{support_size}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        #* save predictions
        preds_dir = self._run_dir / 'predictions_eval'
        preds_dir.mkdir(exist_ok=True, parents=True)
        np.savez(str(preds_dir / f'preds_ds-{ds_filename.replace("/","_")}_support-{support_size}.npz'),
                 output=y_new,
                 prediction_no_finetune=y_sim_new,
                 prediction_jfr_adapted=y_lin_new)
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
        results_mse = {
            'jfr_model': mse_lin.item(),
            'no_finetune': mse_nom.item()
        }
        results_rsquared = {
            'jfr_model': r_sq_lin.item(),
            'no_finetune': r_sq_nom.item()
        }
        results = {'mse': results_mse, 'rsquared': results_rsquared}
        return results


if __name__ == '__main__':

    # model_path = "models/ss_model.pt"
    # BASE_DIR = '/system/user/publicdata/meta_learning/ode/rlc_dataset-resistor_range1_14-capacitor_range100_800-inductor_range20_140'
    # datasets_dir = Path(BASE_DIR) / 'test'

    support_sizes = [10, 20, 30, 50, 70, 100, 2000]

    jfr = JFRTester(experiment_name='jfr_multitask', run_dir=None, seed=0)
    jfr.pretrain()
    jfr.evaluate(split='test',
                 support_sizes=support_sizes,
                 n_datasets=None)
    print('Done.')
