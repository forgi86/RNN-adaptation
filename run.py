
from multitask.cstr_generate import CSTR_Task_Dataset_Gen, CSTR_Task
from multitask.cstr_generate import visualize_tasks



if __name__ == '__main__':

    #
    base_dir = "/home/max/phd/data/ode"
    name = 'cstr'
    n_traj = 2
    n_steps = 256
    num_val_classes = 32
    num_test_classes = 16
    task_factor = 10
    #
    cstr_rand_input = CSTR_Task_Dataset_Gen(base_dir=base_dir,
                              name=name,
                              n_traj=n_traj,
                              n_steps=n_steps,
                              rand_init_state=False,
                              rand_input_u=True,
                              num_val_classes=num_val_classes,
                              num_test_classes=num_test_classes, 
                              task_factor=task_factor)

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
                              task_factor=task_factor)
    cstr_rand_init.save()
