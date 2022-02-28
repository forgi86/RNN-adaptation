import torch
from datetime import datetime
from pathlib import Path

def get_time_str() -> str:
    now = datetime.now()
    year = f'{now.year}'[2:]
    month = f'{now.month}'.zfill(2)
    day = f'{now.day}'.zfill(2)
    hour = f'{now.hour}'.zfill(2)
    minute = f'{now.minute}'.zfill(2)
    second = f'{now.second}'.zfill(2)
    datetime_str = f'{year}{month}{day}_{hour}{minute}{second}'
    return datetime_str

def setup_run_dir(experiment_name: str, run_dir: Path = None) -> Path:
    """Create run directory with unique name. """
    now = datetime.now()
    year = f'{now.year}'[2:]
    month = f'{now.month}'.zfill(2)
    day = f'{now.day}'.zfill(2)
    hour = f'{now.hour}'.zfill(2)
    minute = f'{now.minute}'.zfill(2)
    second = f'{now.second}'.zfill(2)
    run_name = f'{experiment_name}_{year}{month}{day}_{hour}{minute}{second}'

    if run_dir is None:
        run_dir = Path().cwd() / 'runs' / run_name
    else:
        run_dir = run_dir

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


class StateSpaceWrapper(torch.nn.Module):

    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = torch.zeros(2)
        x_sim_torch = self.model(x_0, u_in)
        y_out = x_sim_torch[:, [0]]
        return y_out