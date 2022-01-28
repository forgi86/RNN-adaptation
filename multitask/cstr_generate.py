import copy
import sys
from typing import Any, Dict, List, Union
# sys.path.append("..")
import os
import numpy as np
from worlds.worlds import CSTR
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


class CSTR_Task(object):
    def __init__(self,
                 model_params: Dict[str, Any],
                 n_traj: int = 64,
                 n_steps: int = 256,
                 rand_input_u: bool = True,
                 rand_init_state: bool = False,
                 seed: int = 42,
                 standardize_data: bool = True):
        """Generates a CSTR dataset, for given system parameters

        Args:
            model_params (Dict[str, Any]): the system parameters
            n_traj (int, optional): [description]. Defaults to 64.
            n_steps (int, optional): [description]. Defaults to 256.
            rand_input_u (bool, optional): If true, use (different) random inputs for the `n_traj` trajectories. 
                If false, use the same input for every trajectoryDefaults to True.
            rand_init_state (bool, optional): If true, use (different) random initial states for the `n_traj` trajectories.. Defaults to False.
            seed (int, optional): the seed. Defaults to 42.
            standardize_data (bool, optional): Standardize data. Defaults to True.
        """
        self.n_traj = n_traj
        self.n_steps = n_steps
        self.model_params = model_params
        self.rand_input_u = rand_input_u
        self.rand_init_state = rand_init_state
        self.name_template = 'CA0:{C_A0}_k0list:{k0_list}_Elist:{E_list}'
        self.seed = seed
        self.standardize_data = standardize_data

        self._train_data = self._generate_task_data()

    def _generate_task_data(self):
        """Generates trajectories from CSTR data."""

        system_cstr = CSTR(params=self.model_params)
        input_u, output_y = system_cstr.generate_train_data(
            self.n_traj,
            n_steps=self.n_steps,
            rand_input_u=self.rand_input_u,
            rand_init_state=self.rand_init_state,
            seed=self.seed)
        train_data = np.concatenate((input_u, output_y), axis=2)
        if self.standardize_data:
            data_mean = np.mean(np.mean(train_data, axis=1), axis=0)
            data_std = np.mean(np.std(train_data, axis=1), axis=0)

            # Rescale variables
            train_data = (train_data - data_mean) / data_std

        return train_data

    def save(self, save_dir: Union[str, Path]):
        path = str(save_dir) + '/' + self.filename
        np.save(path, self.train_data)

    @property
    def filename(self) -> str:
        suffix = ''
        # if self.rand_input_u:
        #     suffix = '_randinput'
        return self.name_template.format(**self.model_params).replace(
            ' ', '').replace('[', '').replace(']', '').replace(
                ',', '-') + suffix + '.npy'

    @property
    def train_data(self) -> np.ndarray:
        return self._train_data


class CSTR_Task_Dataset_Gen(object):
    def __init__(self,
                 base_dir: str,
                 name: str,
                 model_params_nominal: Dict[str, Any] = None,
                 n_traj: int = 64,
                 n_steps: int = 256,
                 rand_input_u: bool = True,
                 rand_init_state: bool = False,
                 num_val_classes: int = 512,
                 num_test_classes: int = 256,
                 seed: int = 42,
                 task_factor: int = 10,
                 n_jobs: int = 6):
        self.base_dir = base_dir
        self.name = name
        self.n_traj = n_traj
        self.n_steps = n_steps
        self.rand_input_u = rand_input_u
        self.rand_init_state = rand_init_state
        self.task_factor = task_factor  # this determines the scale (factor) which is multiplied with nominal parameters
        self.n_jobs = n_jobs

        self.model_params_nominal = model_params_nominal
        if self.model_params_nominal is None:
            self.model_params_nominal = {
                'C_A0': 0.8,
                'k0_list': [1.0, 0.7, 0.1, 0.006],
                'E_list': [8.33, 10.0, 50.0, 83.3]
            }

        self.rng = np.random.default_rng(seed)

        self.datasets = {}

        print("Generating datasets...")
        print("Generate train dataset:")
        self.datasets['train'] = self.generate_tasks(
            [self.model_params_nominal])

        print("Generate val datasets:")
        val_params = self.generate_task_params(num_val_classes)
        self.datasets['val'] = self.generate_tasks(val_params)

        print("Generate test datasets:")
        test_params = self.generate_task_params(num_test_classes)
        self.datasets['test'] = self.generate_tasks(test_params)

    def generate_task_params(self, n_tasks: int = 100) -> List[Dict[str, Any]]:
        """Samples a scaling factor for the model parameters C_A0 and for k0_list."""

        params = []

        scale_c_a0 = self.rng.uniform(1., self.task_factor, size=n_tasks)
        scale_k0_list = self.rng.uniform(1., self.task_factor, size=n_tasks)

        for i in range(n_tasks):
            # TODO generate tasks differently here
            task_params = copy.deepcopy(self.model_params_nominal)
            task_params['C_A0'] *= scale_c_a0[i]
            task_params['k0_list'] = (np.array(task_params['k0_list']) *
                                      scale_k0_list[i]).tolist()

            params.append(task_params)

        return params

    def generate_tasks(self, param_list: List[Dict[str,
                                                   Any]]) -> List[CSTR_Task]:
        tasks = []

        def create_task(params):
            seed = self.rng.integers(low=0, high=1000000)
            task = CSTR_Task(params,
                             self.n_traj,
                             self.n_steps,
                             self.rand_input_u,
                             self.rand_init_state,
                             seed=seed,
                             standardize_data=True)

            return task

        tasks = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(create_task)(params)
            for params in tqdm(param_list, desc="Generating datasets"))

        # single thread
        # for params in tqdm(param_list, desc="Generating datasets"):
        #     task = create_task(params)
        #     tasks.append(task)
        return tasks

    def visualize_model_variation(self):
        tasks = []
        for i in range(1, 11):
            new_params = copy.deepcopy(self.model_params_nominal)
            new_params['C_A0'] *= i
            e_array = np.array(new_params['k0_list']) * i 
            new_params['k0_list'] = e_array.tolist()
            tasks.append(
                CSTR_Task(new_params,
                          n_traj=1,
                          rand_input_u=False,
                          rand_init_state=False,
                          seed=42))

        return visualize_tasks(tasks, title='model variation range')

    def save(self, save_nom_pert_diff=True):
        suffix = ''
        if self.rand_init_state:
            suffix += '_randinit'
        if self.rand_input_u:
            suffix += '_randinput'

        name = self.name + f"_ntraj{self.n_traj}_nsteps{self.n_steps}_taskfactor{self.task_factor}" + suffix
        base_dir = Path(self.base_dir) / name

        for split, datasets in self.datasets.items():
            split_dir = base_dir / split
            split_dir.mkdir(exist_ok=True, parents=True)
            for ds in datasets:
                ds.save(split_dir)
            # save dataset visualization
            traj_idx = 0
            fig = visualize_tasks(
                datasets,
                traj_index=traj_idx,
                title=f"Tasks '{split}' - traj_idx: {traj_idx}")
            figname = str(base_dir) + f'/{split}_tasks.jpg'
            fig.savefig(figname, dpi=300, bbox_inches="tight")

        # add-on
        if save_nom_pert_diff:
            fig = visualize_nominal_perturbed()
            figname = str(base_dir) + f'/nom_vs_perturbed.jpg'
            fig.savefig(figname, dpi=300, bbox_inches="tight")


def visualize_tasks(tasks_data: List[CSTR_Task],
                    traj_index: int = 0,
                    title: str = None):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    if title:
        plt.suptitle(title + f" - num_tasks: {len(tasks_data)}")
    for task in tasks_data:
        data = task.train_data
        ax[0].plot(data[traj_index, :, 0])  # q
        ax[0].set_ylabel('input q')
        ax[0].grid(True)
        ax[1].plot(data[traj_index, :, 1])  # T
        ax[1].set_ylabel('input T')
        ax[1].grid(True)
        ax[2].plot(data[traj_index, :, 2])  # Ca
        ax[2].set_ylabel('output Ca')
        ax[2].grid(True)
        ax[3].plot(data[traj_index, :, 3], label=task.filename)  # Cr
        ax[3].set_ylabel('output Cr')
        ax[3].set_xlabel('n_steps / Ts=0.1')
        ax[3].grid(True)
        ax[3].legend(frameon=False, loc=(1, 0))

    return fig


def visualize_nominal_perturbed():
    # parameters from paper
    params_nom = {
        'C_A0': 0.8,
        'k0_list': [1.0, 0.7, 0.1, 0.006],
        'E_list': [8.33, 10.0, 50.0, 83.3]
    }

    params_pert = {
        'C_A0': 0.75,
        'k0_list': [1.0, 0.7, 0.1, 0.006],
        'E_list': [7.33, 9.0, 60.0, 93.3]
    }

    task_nom = CSTR_Task(params_nom,
                         rand_input_u=False,
                         rand_init_state=False,
                         seed=42)
    task_pert = CSTR_Task(params_pert,
                          rand_input_u=False,
                          rand_init_state=False,
                          seed=42)

    fig = visualize_tasks([task_nom, task_pert],
                          title='nominal vs. perturbed system')
    return fig