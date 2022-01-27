from random import seed
import sys
from typing import Any, Dict, List, Union

# from ray import data
# from zmq import frame
# sys.path.append("..")
import os
import numpy as np
from worlds.worlds import CSTR
import matplotlib.pyplot as plt


class CSTR_Task(object):
    def __init__(self,
                 model_params: Dict[str, Any],
                 n_traj: int = 64,
                 n_steps: int = 256,
                 rand_input_u: bool = True,
                 rand_init_state: bool = False,
                 seed: int = 42,
                 standardize_data: bool = True):
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

    @property
    def filename(self) -> str:
        suffix = ''
        if self.rand_input_u:
            suffix = '_randinput'
        return self.name_template.format(**self.model_params).replace(
            ' ', '').replace('[', '').replace(']', '').replace(
                ',', '-') + suffix + '.npy'

    @property
    def train_data(self) -> np.ndarray:
        return self._train_data


class CSTR_Task_Dataset(object):
    def __init__(self, base_dir: str, model_params_lower, model_params_lower):
        pass

    # parameter generation
    def _sample_model_params():
        pass

    def _model_params_range():
        pass


def visualize_tasks(tasks_data: List[CSTR_Task], traj_index: int = 0):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

    for task in tasks_data:
        data = task.train_data
        ax[0].plot(data[traj_index, :, 0], label=task.filename)  # q
        ax[1].plot(data[traj_index, :, 1])  # T
        ax[2].plot(data[traj_index, :, 2])  # Ca
        ax[3].plot(data[traj_index, :, 3])  # Cr
        ax[0].legend(frameon=False, loc=(1, 0))

    return fig
