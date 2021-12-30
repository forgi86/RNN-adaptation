import os
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import torch


class PlotDyno():
    def __init__(self):
        # Parameters
        self.batch_size = 1
        self.context = 25

        # Import test and train values
        self.u_train = torch.tensor(np.load(os.path.join("data", "cstr", "u_train.npy")).astype(np.float32))
        self.y_train = torch.tensor(np.load(os.path.join("data", "cstr", "y_train.npy")).astype(np.float32))
        self.u_test = torch.tensor(np.load(os.path.join("data", "cstr", "u_test.npy")).astype(np.float32))
        self.y_test = torch.tensor(np.load(os.path.join("data", "cstr", "y_test.npy")).astype(np.float32))
        self.u_transf = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[:self.batch_size, :, :]
        self.y_transf = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[:self.batch_size, :, :]

        # Import evaluation data # Shape: (batch_size, seq_len, input_size/ output_size)
        self.u_eval = np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)[:self.batch_size, :, :]
        self.y_eval = np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)[:self.batch_size, :, :]

        self.seq_len = self.u_eval.shape[1]
        self.y_eval_0 = self.y_eval[..., [0]]  # Single output
        self.y_eval_1 = self.y_eval[..., [1]]  # Single output

        self.y_torch_new_0 = torch.tensor(self.y_eval_0[:, 1:, :].reshape(-1, 1), dtype=torch.float)  # Single output
        self.y_context_0 = self.y_torch_new_0[1:self.context+1, :].detach().numpy()
        self.y_torch_new_1 = torch.tensor(self.y_eval_1[:, 1:, :].reshape(-1, 1), dtype=torch.float)  # Single output
        self.y_context_1 = self.y_torch_new_1[1:self.context+1, :].detach().numpy()

        # Import evaluation results from parametric approach
        self.y_sim_02 = np.load(os.path.join("data", "cstr", "02_cstr_eval.npy")).astype(np.float32)

        # Import evaluation results after linearization
        self.y_sim_04 = np.load(os.path.join("data", "cstr", "04_cstr_eval_sim.npy")).astype(np.float32)
        self.y_lin_04 = np.load(os.path.join("data", "cstr", "04_cstr_eval_lin.npy")).astype(np.float32)

        # Import evaluation results from Jacobian-vector product
        self.y_sim_gd = np.load(os.path.join("data", "cstr", "cstr_eval_gd_sim.npy")).astype(np.float32)
        self.y_lin_gd = np.load(os.path.join("data", "cstr", "cstr_eval_gd_lin.npy")).astype(np.float32)

        # Import evaluation results from non-parametric approach
        self.y_lin_gp_0 = np.load(os.path.join("data", "cstr", "GP_predict_0.npy")).astype(np.float32)
        self.upper_conf_0 = np.load(os.path.join("data", "cstr", "GP_upper_conf_0.npy")).astype(np.float32)
        self.lower_conf_0 = np.load(os.path.join("data", "cstr", "GP_lower_conf_0.npy")).astype(np.float32)

        self.y_lin_gp_1 = np.load(os.path.join("data", "cstr", "GP_predict_1.npy")).astype(np.float32)
        self.upper_conf_1 = np.load(os.path.join("data", "cstr", "GP_upper_conf_1.npy")).astype(np.float32)
        self.lower_conf_1 = np.load(os.path.join("data", "cstr", "GP_lower_conf_1.npy")).astype(np.float32)


    def plot_algo_compare(self):
        # Matplotlib
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.y_eval[0, 1:, 0], 'k')
        ax[0].plot(np.concatenate((self.y_context_0[:, 0], self.y_sim_02[0, self.context:, 0]), axis=0), 'b')
        # ax[0].plot(y_sim_02[0, :, 0], 'b', label="LSTM-02")
        # ax[0].plot(y_sim_04[:, 0], 'y', label="LSTM-04")
        ax[0].plot(np.concatenate((self.y_context_0[:, 0], self.y_lin_04[self.context:, 0]), axis=0), '--r')
        # ax[0].plot(y_sim_gd[:, 0], 'm', label="LSTM-GD")
        ax[0].plot(np.concatenate((self.y_context_0[:, 0], self.y_lin_gd[self.context:, 0]), axis=0),
                   '--c')  # Jacobian-Vector Product
        ax[0].plot(np.concatenate((self.y_context_0[:, 0], self.y_lin_gp_0[:, 0]), axis=0), '--y')
        ax[0].axvline(self.context - 1, color='k', linestyle='--', alpha=0.2)
        ax[0].set_ylabel('Y')
        ax[0].set_xlabel('X')
        ax[0].grid(True)

        ax[1].plot(self.y_eval[0, 1:, 1], 'k', label="Ground truth")
        ax[1].plot(np.concatenate((self.y_context_1[:, 0], self.y_sim_02[0, self.context:, 1]), axis=0), 'b',
                   label="LSTM")
        # ax[1].plot(y_sim_02[0, :, 1], 'b')
        # ax[1].plot(y_sim_04[:, 1], 'y')
        ax[1].plot(np.concatenate((self.y_context_1[:, 0], self.y_lin_04[self.context:, 1]), axis=0), '--r',
                   label="BLR-LSTM")
        # ax[1].plot(y_sim_gd[:, 1], 'm')
        ax[1].plot(np.concatenate((self.y_context_1[:, 0], self.y_lin_gd[self.context:, 1]), axis=0), '--c',
                   label="JVP-LSTM")
        ax[1].plot(np.concatenate((self.y_context_1[:, 0], self.y_lin_gp_1[:, 0]), axis=0), '--y', label="GP-LSTM")
        ax[1].axvline(self.context - 1, color='k', linestyle='--', alpha=0.2)
        ax[1].set_ylabel('Y')
        ax[1].set_xlabel('X')
        ax[1].legend()
        ax[1].grid(True)

        fig.tight_layout()
        plt.savefig('compare.png', dpi=300)

    def plot_algo_compare_sns(self):
        x = np.arange(self.seq_len-1)

        data_op0 = np.vstack([x,
                              self.y_eval[0, 1:, 0],  # 1023
                              np.concatenate((self.y_context_0[:, 0], self.y_sim_02[0, self.context:, 0]), axis=0),
                              np.concatenate((self.y_context_0[:, 0], self.y_lin_04[self.context:, 0]), axis=0),
                              np.concatenate((self.y_context_0[:, 0], self.y_lin_gd[self.context:, 0]), axis=0),
                              np.concatenate((self.y_context_0[:, 0], self.y_lin_gp_0[:, 0]), axis=0)])

        data_op1 = np.vstack([x,
                              self.y_eval[0, 1:, 1],
                              np.concatenate((self.y_context_1[:, 0], self.y_sim_02[0, self.context:, 1]), axis=0),
                              np.concatenate((self.y_context_1[:, 0], self.y_lin_04[self.context:, 1]), axis=0),
                              np.concatenate((self.y_context_1[:, 0], self.y_lin_gd[self.context:, 1]), axis=0),
                              np.concatenate((self.y_context_1[:, 0], self.y_lin_gp_1[:, 0]), axis=0)])

        print(data_op1.shape, data_op0.shape)
        y0 = pd.DataFrame(np.transpose(data_op0))
        y1 = pd.DataFrame(np.transpose(data_op1), columns=["Samples", "Ground truth", "LSTM",
                                                           "BLR", "LM-BLR", "GP-LSTM"])

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 1)
        sns.lineplot(ax=axes[0], data=y0.iloc[:, 1:])
        axes[0].legend([], [], frameon=False)
        axes[0].set(ylabel=r'$\mathrm{C_A}$')
        axes[0].axvline(self.context - 1, color='k', linestyle='--', alpha=0.2)
        sns.lineplot(ax=axes[1], data=y1.iloc[:, 1:])
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend([], [], frameon=False)
        axes[1].axvline(self.context - 1, color='k', linestyle='--', alpha=0.2)
        axes[1].legend(handles, labels, ncol=5, loc='lower center', frameon=False, fontsize=8)
        axes[1].set(xlabel=r'$\mathrm{Samples}$', ylabel=r'$\mathrm{C_R}$')
        fig.tight_layout()
        plt.savefig('compare_sns.png', dpi=300)

    def compare_input_seq(self):
        print("Shapes: ", self.u_train.shape, self.u_test.shape, self.u_transf.shape, self.u_eval.shape)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.u_train[0, :, 0], label="Train")
        ax[0].plot(self.u_test[0, :, 0], label="Test")
        ax[0].plot(self.u_transf[0, :, 0], label="Transfer")
        ax[0].plot(self.u_eval[0, :, 0], label="Evaluate")
        ax[0].set_ylabel('Temperature (T)')
        ax[0].set_xlabel('Samples')
        ax[0].grid(True)

        ax[1].plot(self.u_train[0, :, 1], label="Train")
        ax[1].plot(self.u_test[0, :, 1], label="Test")
        ax[1].plot(self.u_transf[0, :, 1], label="Transfer")
        ax[1].plot(self.u_eval[0, :, 1], label="Evaluate")
        ax[1].set_xlabel('Samples')
        ax[1].set_ylabel('Feed Rate (q)')
        ax[1].legend(loc='lower center', ncol=4)
        ax[1].grid(True)
        fig.tight_layout()

    def compare_output_seq(self):
        print("Shapes: ", self.y_train.shape, self.y_test.shape, self.y_transf.shape, self.y_eval.shape)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.y_train[0, :, 0], label="Train")
        ax[0].plot(self.y_test[0, :, 0], label="Test")
        ax[0].plot(self.y_transf[0, :, 0], label="Transfer")
        ax[0].plot(self.y_eval[0, :, 0], label="Evaluate")
        ax[0].set_ylabel(r'$\mathrm{C_A}$')
        ax[0].set_xlabel('Samples')
        ax[0].grid(True)

        ax[1].plot(self.y_train[0, :, 1], label="Train")
        ax[1].plot(self.y_test[0, :, 1], label="Test")
        ax[1].plot(self.y_transf[0, :, 1], label="Transfer")
        ax[1].plot(self.y_eval[0, :, 1], label="Evaluate")
        ax[1].set_xlabel('Samples')
        ax[1].set_ylabel(r'$\mathrm{C_R}$')
        ax[1].legend(loc='lower center', ncol=4)
        ax[1].grid(True)
        fig.tight_layout()


if __name__ == "__main__":
    pt = PlotDyno()
    # pt.plot_algo_compare_sns()
    pt.plot_algo_compare_sns()
    plt.show()

