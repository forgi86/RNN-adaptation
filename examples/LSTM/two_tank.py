from worlds.worlds import Two_tank_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    system = Two_tank_model()

    batch_size = 2
    n_steps = 1000
    input_train, input_test, output_train, output_test, init_states_train, init_states_test = system.generate_data(batch_size, n_steps)

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(input_train[0, :, 0])  # q
    #ax[1].plot(input_train[0, :, 1])  # T
    ax[1].plot(output_train[0, :, 0])  # Ca
    #ax[3].plot(output_train[0, :, 1])  # Cr
