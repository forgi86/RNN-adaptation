import argparse
from multitask.cstr_generate import CSTR_Task_Dataset_Gen, CSTR_Task
from multitask.cstr_generate import visualize_tasks

# command:
# python run_dataset_gen.py --base-dir /home/max/phd/data/ode --name cstr --n-traj 2 --n-steps 256 --num-val-classes 16 --num-test-classes 8 --task-factor 10


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--name', type=str, default='cstr')
    parser.add_argument('--n-traj', type=int, default=256)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--num-val-classes', type=int, default=256)
    parser.add_argument('--num-test-classes', type=int, default=256)
    parser.add_argument('--task-factor', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=6)
    args = vars(parser.parse_args())
    return args


def _main():
    args = _get_args()
    #
    base_dir = args['base_dir']
    name = args['name']
    n_traj = args['n_traj']
    n_steps = args['n_steps']
    num_val_classes = args['num_val_classes']
    num_test_classes = args['num_test_classes']
    task_factor = args['task_factor']
    n_jobs = args['n_jobs']
    #
    cstr_rand_input = CSTR_Task_Dataset_Gen(base_dir=base_dir,
                                            name=name,
                                            n_traj=n_traj,
                                            n_steps=n_steps,
                                            rand_init_state=False,
                                            rand_input_u=True,
                                            num_val_classes=num_val_classes,
                                            num_test_classes=num_test_classes,
                                            task_factor=task_factor,
                                            n_jobs=n_jobs)

    cstr_rand_input.save()
    #
    cstr_rand_init = CSTR_Task_Dataset_Gen(base_dir=base_dir,
                                           name=name,
                                           n_traj=n_traj,
                                           n_steps=n_steps,
                                           rand_init_state=True,
                                           rand_input_u=False,
                                           num_val_classes=num_val_classes,
                                           num_test_classes=num_test_classes,
                                           task_factor=task_factor,
                                           n_jobs=n_jobs)
    cstr_rand_init.save()


if __name__ == '__main__':
    _main()
