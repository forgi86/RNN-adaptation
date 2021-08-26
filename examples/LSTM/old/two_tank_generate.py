from worlds.worlds import Two_tank_model
import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch_size = 2
    T = 120000  # simulation length (s)
    Ts = 10  # sampling time (s)
    n_steps = T//Ts

    system = Two_tank_model(Ts=Ts)

    input_train, input_test, output_train, output_test, init_states_train, init_states_test\
        = system.generate_data(batch_size, n_steps)

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(input_train[0, :, 0])  # q
    #ax[1].plot(input_train[0, :, 1])  # T
    ax[1].plot(output_train[0, :, 0])  # Ca
    #ax[3].plot(output_train[0, :, 1])  # Cr
