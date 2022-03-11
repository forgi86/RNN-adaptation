import copy
import enum
from random import sample
from statistics import mode
import sys
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import control.matlab
from examples.RLC.symbolic_RLC import fxu_ODE, fxu_ODE_nl
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class RLC_Task(object):

    def __init__(
            self,
            model_params: Dict[str, Any],
            n_traj: int = 6,  # prev.: 32
            n_steps: int = 2000,
            Ts: float = 1e-6,  # in s
            input_bandwidth: float = 80e3,  # in Hz
            scale_u: float = 80.,  # in V
            scale_x: List[float] = [90., 3.],
            non_linear: bool = True,
            x0: List[float] = [0., 0.]):

        self.model_params = model_params
        self.n_traj = n_traj
        self.n_steps = n_steps

        self.Ts = Ts
        self.input_bandwidth = input_bandwidth
        self.scale_u = scale_u
        self.scale_x = scale_x
        self.non_linear = non_linear
        if x0:
            self.x0 = np.array(x0)
        else:
            self.x0 = np.zeros(2)

        self.name_template = 'R:{R}_L:{L}_C:{C}'

        self.rng = np.random.default_rng()

        # data has shape (n_traj, n_steps, n_dims)
        # n_dims = len([te, u, v_c, i_l]) = 4
        self.data = self._generate_task_data()

    def _generate_task_data(self) -> np.ndarray:

        # linear vs non-linear variant
        ode = fxu_ODE_nl if self.non_linear else fxu_ODE

        def f_ODE(t, x):
            u = u_func(t).ravel()
            return ode(t, x, u, params=self.model_params)

        trajectories = []

        for i in range(self.n_traj):

            u, te = self._generate_input_time()
            u_func = interp1d(te, u, kind='zero', fill_value="extrapolate")

            t_span = (te[0], te[-1])
            y = solve_ivp(f_ODE, t_span, self.x0, t_eval=te)  # Linear

            # system states [v_c, i_l]
            x = y.y.T  # transpose -> x has shape (n_steps, 2)

            traj = np.concatenate((te.reshape(-1, 1), u.reshape(-1, 1), x),
                                  axis=1)
            trajectories.append(traj)

        return np.array(trajectories)

    def _generate_input_time(self) -> Tuple[np.ndarray, np.ndarray]:
        # white noise is a random process
        # white noise has all frequencies -> not possible -> need filter

        # input transfer function -> this is the filter used to filter the white noise
        Hu = control.TransferFunction([1], [1 / self.input_bandwidth, 1])
        Hu = Hu * Hu
        # discretize the filter < through this filter the random process is passed
        # Hud = control.matlab.c2d(Hu, self.Ts)

        # generate the white noise
        e = self.rng.standard_normal(self.n_steps)
        te = np.arange(self.n_steps) * self.Ts

        # filter the white noise (-> colored noise)
        _, u = control.forced_response(Hu, te, e, return_x=False)

        # scale input signal
        u = u / np.std(u) * self.scale_u

        return u, te

    def save(self, save_dir: Union[str, Path]):
        path = str(save_dir) + '/' + self.filename
        np.save(path, self.data)

    @property
    def filename(self) -> str:
        # prefix = 'nonlin_' if self.non_linear else 'lin_'
        prefix = ''
        return prefix + self.name_template.format(**self.model_params) + '.npy'


class RLC_Task_Dataset_Gen(object):

    def __init__(
            self,
            base_dir: str,
            name: str = 'rlc_dataset',
            model_params_nominal: Dict[str, Any] = {
                "C": 270e-9,
                "L": 50e-6,
                "R": 3.0
            },
            num_val_classes: int = 512,
            num_test_classes: int = 256,
            seed: int = 1,
            resistor_range=(1, 14),  # in Ohm
            inductor_range=(20, 140),  # in µH
            capacitor_range=(100, 800),  # in nF
            n_jobs=24):
        self.base_dir = base_dir
        self.name = name
        self.model_params_nominal = model_params_nominal
        self.n_jobs = n_jobs
        self.resistor_range = resistor_range
        self.inductor_range = inductor_range
        self.capacitor_range = capacitor_range

        np.random.seed(seed)

        self.datasets = {}

        print("Generating datasets...")
        print("Generate train dataset:")
        self.datasets['train'] = self.generate_tasks(
            [self.model_params_nominal])

        print("Generate val datasets:")
        val_params = self.generate_task_params(num_val_classes, resistor_range,
                                               inductor_range, capacitor_range)
        self.datasets['val'] = self.generate_tasks(val_params)

        print("Generate test datasets:")
        test_params = self.generate_task_params(num_test_classes,
                                                resistor_range, inductor_range,
                                                capacitor_range)
        self.datasets['test'] = self.generate_tasks(test_params)

    def generate_task_params(
            self,
            n_tasks: int = 100,
            resistor_range=(2, 7),  # in Ohm
            inductor_range=(40, 70),  # in µH
            capacitor_range=(200, 400),  # in nF
    ) -> List[Dict[str, Any]]:
        rng = np.random.default_rng()

        def sample_param_dict():
            params = {}
            r_val = rng.uniform(low=resistor_range[0], high=resistor_range[1])
            l_val = rng.uniform(low=inductor_range[0], high=inductor_range[1])
            c_val = rng.uniform(low=capacitor_range[0],
                                high=capacitor_range[1])

            params['R'] = r_val
            params['L'] = l_val * 1e-6
            params['C'] = c_val * 1e-9

            return params

        param_list = []
        for i in range(n_tasks):
            p = sample_param_dict()
            param_list.append(p)

        return param_list

    def generate_tasks(self, param_list: List[Dict[str,
                                                   Any]]) -> List[RLC_Task]:
        tasks = []

        def create_task(params):
            task = RLC_Task(model_params=params)
            return task

        tasks = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(create_task)(params)
            for params in tqdm(param_list, desc="Generating datasets"))

        return tasks

    def save(self):
        suffix = f"-resistor_range{self.resistor_range}-capacitor_range{self.capacitor_range}-inductor_range{self.inductor_range}"
        suffix = suffix.replace('(',
                                '').replace(')',
                                            '').replace(' ',
                                                        '').replace(',', '_')
        name = self.name + suffix
        base_dir = Path(self.base_dir) / name
        print("Saving datasets..")
        for split, datasets in self.datasets.items():
            split_dir = base_dir / split
            split_dir.mkdir(exist_ok=True, parents=True)
            for ds in datasets:
                ds.save(split_dir)

        print("Visualizing tasks ...")
        n_vis_tasks = 100
        fig = visualize_tasks(n_vis_tasks, self.resistor_range,
                              self.inductor_range, self.capacitor_range)
        # figname = str(
        #     base_dir
        # ) + f"/{n_vis_tasks} tasks - resistor_range {self.resistor_range} Ohm, \ncapacitor_range {self.capacitor_range} nF, \ninductor_range {self.inductor_range} µH"
        figname = str(
            base_dir
        ) + f"/{n_vis_tasks} tasks - resistor_range {self.resistor_range} Ohm, \ncapacitor_range {self.capacitor_range} nF, \ninductor_range {self.inductor_range} µH.png"

        fig.savefig(figname, dpi=300, bbox_inches="tight")


def generate_colored_noise(bandwidth: float,
                           Ts: float,
                           n_steps: int,
                           scale_signal: float = 1.0,
                           seed: int = 1):
    rng = np.random.default_rng(seed=seed)
    # white noise is a random process
    # white noise has all frequencies -> not possible -> need filter
    # input transfer function -> this is the filter used to filter the white noise
    Hu = control.TransferFunction([1], [1 / bandwidth, 1])
    Hu = Hu * Hu
    # discretize the filter < through this filter the random process is passed
    # Hud = control.matlab.c2d(Hu, self.Ts)
    # generate the white noise
    e = rng.standard_normal(n_steps)
    te = np.arange(n_steps) * Ts
    # filter the white noise (-> colored noise)
    _, u = control.forced_response(Hu, te, e, return_x=False)
    # scale input signal
    u = u / np.std(u) * scale_signal
    return u, te


def generate_rlc_task_params(
        n_tasks: int,
        seed: int = 1,
        resistor_range=(2, 7),  # in Ohm
        inductor_range=(40, 70),  # in µH
        capacitor_range=(200, 400),  # in nF
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed=seed)

    def sample_param_dict():
        params = {}
        r_val = rng.uniform(low=resistor_range[0], high=resistor_range[1])
        l_val = rng.uniform(low=inductor_range[0], high=inductor_range[1])
        c_val = rng.uniform(low=capacitor_range[0], high=capacitor_range[1])
        params['R'] = r_val
        params['L'] = l_val * 1e-6
        params['C'] = c_val * 1e-9

        return params

    param_list = []
    for i in range(n_tasks):
        p = sample_param_dict()
        param_list.append(p)
    return param_list


def simulate_rlc_system(model_params: Dict[str, Any],
                        u: np.ndarray,
                        te: np.ndarray,
                        non_linear: bool = True) -> np.ndarray:
    # linear vs non-linear variant
    ode = fxu_ODE_nl if non_linear else fxu_ODE

    def f_ODE(t, x):
        u = u_func(t).ravel()
        return ode(t, x, u, params=model_params)

    x0 = np.zeros(2)

    u_func = interp1d(te, u, kind='zero', fill_value="extrapolate")
    t_span = (te[0], te[-1])
    y = solve_ivp(f_ODE, t_span, x0, t_eval=te)  # Linear
    # system states [v_c, i_l]
    x = y.y.T  # transpose -> x has shape (n_steps, 2)

    return x

def visualize_tasks(
        n_tasks=100,
        task_params: List[Dict[str, float]] = None,
        resistor_range=(2, 7),  # in Ohm
        inductor_range=(40, 70),  # in µH
        capacitor_range=(200, 400),  # in nF
        plot_nominal_model: bool= False,
        n_steps:int = 2000
):
    # inputs
    u, te = generate_colored_noise(bandwidth=80e3, Ts=1e-6, n_steps=n_steps)
    if task_params:
        task_models = task_params
    else:
        # different models for the tasks
        task_models = generate_rlc_task_params(n_tasks,
                                               resistor_range=resistor_range,
                                               inductor_range=inductor_range,
                                               capacitor_range=capacitor_range)
    
    if plot_nominal_model:
        nominal_model = {"C": 270e-9, "L": 50e-6, "R": 3.0}
        # simulate systems
        nominal_traj = simulate_rlc_system(nominal_model, u, te)

    task_trajectories = []
    for task_model in task_models[:n_tasks]:
        traj = simulate_rlc_system(task_model, u, te)
        task_trajectories.append(traj)

    name_template = 'R:{R}_L:{L}_C:{C}'

    # plot trajectories
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plot_steps = n_steps
    for i, task_traj in enumerate(task_trajectories):
        ax[0].plot(te[:plot_steps],
                   task_traj[:plot_steps, 0],
                   label=name_template.format(**task_models[i]))
        ax[1].plot(te[:plot_steps],
                   task_traj[:plot_steps, 1],
                   label=name_template.format(**task_models[i]))
    if plot_nominal_model:
        ax[0].plot(te[:plot_steps],
                   nominal_traj[:plot_steps, 0],
                   'r',
                   lw=2,
                   label=name_template.format(**nominal_model))
        ax[1].plot(te[:plot_steps],
                   nominal_traj[:plot_steps, 1],
                   'r',
                   lw=2,
                   label=name_template.format(**nominal_model))
    ax[0].legend(frameon=False, loc=(1, 0))

    ax[2].plot(te[:plot_steps], u[:plot_steps], 'b')
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('Input voltage (V)')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Inductor current (A)')
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Capacitor voltage (V)')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.suptitle(
        f"{n_tasks} tasks - resistor_range {resistor_range} Ohm, \ncapacitor_range {capacitor_range} nF, \ninductor_range {inductor_range} µH"
    )
    return fig


if __name__ == '__main__':

    # params_perturbed = {"C": 350e-9, "L": 50e-6, "R": 4.0}
    # params = {"C": 270e-9, "L": 50e-6, "R": 3.0}
    # task = RLC_Task(model_params=params, n_traj=3)
    # print('generated')

    # fig = visualize_tasks()
    # plt.show()
    # figname = 'rlc_tasks.jpg'
    # fig.savefig(figname, dpi=300, bbox_inches="tight")

    # # base_dir = "home/max/phd/data/ode"

    # rlc_data = RLC_Task_Dataset_Gen(base_dir=base_dir)
    # rlc_data.save()
    # print("Done.")

    # Generate datasets from Forgione Paper
    params_transfereval = {"C": 350e-9, "L": 50e-6, "R": 4.0}
    # params = {"C": 270e-9, "L": 50e-6, "R": 3.0}
    task_transfereval = RLC_Task(model_params=params_transfereval, n_traj=3)
    base_dir = Path("/system/user/publicdata/meta_learning/ode")
    dataset_dir = base_dir / "rlc_dataset-resistor_range1_14-capacitor_range100_800-inductor_range20_140"

    save_dir_transfereval = dataset_dir / 'transfereval'
    save_dir_transfereval.mkdir(exist_ok=True, parents=True)
    task_transfereval.save(save_dir_transfereval)

    # print('generated')