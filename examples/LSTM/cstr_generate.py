from worlds.worlds import CSTR
import matplotlib.pyplot as plt

if __name__ == "__main__":
    system = CSTR()

    batch_size = 32
    n_steps = 100
    u_train, u_test, y_train, y_test = system.generate_data(batch_size, n_steps, flow_period=50)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(u_train[0, :, 0])  # q
    ax[1].plot(u_train[0, :, 1])  # T
    ax[2].plot(y_train[0, :, 0])  # Ca
    ax[3].plot(y_train[0, :, 1])  # Cr
