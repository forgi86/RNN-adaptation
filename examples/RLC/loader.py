import os
import numpy as np
import pandas as pd

from pathlib import Path

COL_T = ['time']
COL_X = ['V_C', 'I_L']
COL_U = ['V_IN']
COL_Y = ['V_C']

BASE_DIR = '/system/user/publicdata/meta_learning/ode/rlc_dataset-resistor_range1_14-capacitor_range100_800-inductor_range20_140'


def rlc_loader(dataset,
               dataset_type="nl",
               output='V_C',
               noise_std=0.1,
               dtype=np.float32,
               scale=True,
               n_data=-1):
    filename = f"RLC_data_{dataset}_{dataset_type}.csv"
    df_data = pd.read_csv(os.path.join("data", filename))
    t = np.array(df_data[['time']], dtype=dtype)
    u = np.array(df_data[['V_IN']], dtype=dtype)
    y = np.array(df_data[[output]], dtype=dtype)
    x = np.array(df_data[['V_C', 'I_L']], dtype=dtype)
    y += np.random.randn(*y.shape) * noise_std
    if scale:
        u = u / 100
        y = y / 100
        x = x / [100, 6]

    if n_data > 0:
        t = t[:n_data, :]
        u = u[:n_data, :]
        y = y[:n_data, :]
        x = x[:n_data, :]
    return t, u, y, x


if __name__ == "__main__":
    t, u, y, x = rlc_loader("train", "lin")


def rlc_loader_multitask(ds_filename: str,
                         trajectory: int,
                         steps: int,
                         trajectory_stop: int = None, 
                         noise_std: float = 0.1,
                         scale: bool = True,
                         base_dir: str = BASE_DIR, 
                         dtype=np.float32):
    """
    if trajectory_end is given the training data consists of multiple trajectories and 
    return values have one additional trajectory dimension.
    """
    base_dir = Path(base_dir)
    file = base_dir / ds_filename

    # dimensions: (trajectory_sample, step, variable [te, vin, vc, il])
    # e.g. all_data[0,1,2]
    all_data = np.load(file, allow_pickle=True)

    # scale data according to Forgione paper code
    # scale vin (input voltage)
    all_data[:, :, 1] /= 80.
    # scale vc (output voltage)
    all_data[:, :, 2] /= 90.
    # scale il (inductor current)
    all_data[:, :, 3] /= 3.

    # add output noise according to Forgione paper code
    all_data[:, :, 2] += np.random.randn(*all_data[:, :, 2].shape) * noise_std
    if not trajectory_stop:
        all_data_traj = all_data[trajectory]

        # extract t, u, x, y
        t = all_data_traj[:, [0]]
        u = all_data_traj[:, [1]]
        y = all_data_traj[:, [2]]
        x = all_data_traj[:, 2:]

        if scale:
            u = u / 100
            y = y / 100
            x = x / [100, 6]

        if steps > 0:
            t = t[:steps, :]
            u = u[:steps, :]
            y = y[:steps, :]
            x = x[:steps, :]
    else:
        assert trajectory_stop > trajectory
        all_data_traj = all_data[trajectory:trajectory_stop]

        # extract t, u, x, y
        t = all_data_traj[:, :, [0]]
        u = all_data_traj[:, :, [1]]
        y = all_data_traj[:, :, [2]]
        x = all_data_traj[:, :, 2:]

        if scale:
            u = u / 100
            y = y / 100
            x = x / [100, 6]

        if steps > 0:
            t = t[:, :steps, :]
            u = u[:, :steps, :]
            y = y[:, :steps, :]
            x = x[:, :steps, :]

    return t.astype(dtype), u.astype(dtype), y.astype(dtype), x.astype(dtype)


if __name__ == '__main__':
    # ds_filename = 'train/R:3.0_L:5e-05_C:2.7e-07.npy'
    ds_filename = 'transfereval/R:4.0_L:5e-05_C:3.5e-07.npy'
    rlc_loader_multitask(ds_filename, 0, 2000)
