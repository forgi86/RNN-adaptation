import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loader import rlc_loader
from torchid import metrics


class PlotRLC():
    def __init__(self):
        # Parameters
        self.batch_size = 1
        self.t, self.u, self.y, self.x = rlc_loader("train", "nl", noise_std=0.1)
        self.ts = self.t[1, 0] - self.t[0, 0]

        # Load training results
        self.x_true_torch_val = np.load(os.path.join("data", "RLC_SS_NL", "01_train_x_true.npy")).astype(np.float32)
        self.x_sim_torch_val = np.load(os.path.join("data", "RLC_SS_NL", "01_train_x_sim.npy")).astype(np.float32)
        self.u_torch_val = np.load(os.path.join("data", "RLC_SS_NL", "01_train_u_val.npy")).astype(np.float32)
        self.x_hidden_fit_np = np.load(os.path.join("data", "RLC_SS_NL", "01_train_x_hidden.npy")).astype(np.float32)

        self.LOSS = np.load(os.path.join("data", "RLC_SS_NL", "01_loss.npy")).astype(np.float32)
        self.LOSS_CONSISTENCY = np.load(os.path.join("data", "RLC_SS_NL", "01_consist_loss.npy")).astype(np.float32)
        self.LOSS_FIT = np.load(os.path.join("data", "RLC_SS_NL", "01_fit_loss.npy")).astype(np.float32)

        # Load testing results
        self.time_val_us = np.load(os.path.join("data", "RLC_SS_NL", "02_test_time_val.npy")).astype(np.float32)
        self.x_true_val = np.load(os.path.join("data", "RLC_SS_NL", "02_test_x_true.npy")).astype(np.float32)
        self.x_sim = np.load(os.path.join("data", "RLC_SS_NL", "02_test_x_sim.npy")).astype(np.float32)

        # Load evaluation results after transfer
        self.y_new = np.load(os.path.join("data", "RLC_SS_NL", "04_eval_y.npy")).astype(np.float32)
        self.y_sim_new = np.load(os.path.join("data", "RLC_SS_NL", "04_eval_y_sim.npy")).astype(np.float32)
        self.y_lin_new = np.load(os.path.join("data", "RLC_SS_NL", "04_eval_y_lin.npy")).astype(np.float32)
        print("shapes: ", self.y_new.shape, self.y_sim_new.shape, self.y_lin_new.shape)

        # Load EKF transfer and evaluation results
        self.y_EKF_true = np.load(os.path.join("data", "RLC_SS_NL", "03_transfer_EKF_y_true.npy")).astype(np.float32)
        self.y_EKF_pred = np.load(os.path.join("data", "RLC_SS_NL", "03_transfer_EKF_y_pred.npy")).astype(np.float32)

        # Load EKF eval and evaluation results
        self.y_EKF_true_eval = np.load(os.path.join("data", "RLC_SS_NL", "04_eval_EKF_y_true.npy")).astype(np.float32)
        self.y_EKF_pred_eval = np.load(os.path.join("data", "RLC_SS_NL", "04_eval_EKF_y_pred.npy")).astype(np.float32)


    def plot_training(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.x_true_torch_val[:, 0], label="Ground truth")
        sns.lineplot(ax=axes, data=self.x_sim_torch_val[:, 0], label="Estimation")
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(xlabel=r'$\mathrm{Samples}$', ylabel=r'$\mathrm{v_C}$')
        axes.legend(handles, labels, ncol=2, loc='lower center', frameon=False, fontsize=8)
        """
        sns.lineplot(ax=axes[1], data=self.x_true_torch_val[:, 1])
        sns.lineplot(ax=axes[1], data=self.x_sim_torch_val[:, 1])
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend([], [], frameon=False)
        axes[1].set(ylabel=r'$\mathrm{v_C}$')
        axes[1].legend(handles, labels, ncol=5, loc='lower center', frameon=False, fontsize=7)
        """
        fig.tight_layout()

    def plot_train_loss(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.LOSS, label="Total loss")
        sns.lineplot(ax=axes, data=self.LOSS_CONSISTENCY, label="Consistency loss")
        sns.lineplot(ax=axes, data=self.LOSS_FIT, label="Fitting loss")
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel='Loss (-)', xlabel='Iterations (-)')
        axes.legend(handles, labels, ncol=3, loc='upper center', frameon=False, fontsize=10)
        fig.tight_layout()

    def plot_test(self):
        t_plot_start = 0.0e-3
        t_plot_end = t_plot_start + 1.0  # From 02_RLC_test.py
        idx_plot_start = int(t_plot_start // self.ts)
        idx_plot_end = int(t_plot_end // self.ts)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        axes.plot(self.time_val_us[idx_plot_start:idx_plot_end],
                  self.x_true_val[idx_plot_start:idx_plot_end, 0], label='$v_C$')
        axes.plot(self.time_val_us[idx_plot_start:idx_plot_end],
                  self.x_sim[idx_plot_start:idx_plot_end, 0],
                  linestyle='dashed', label=r'$\hat{v}^{\mathrm{sim}}_C$')
        axes.legend(loc='upper right')
        axes.grid(True)
        axes.set_xlabel(r'$\mathrm{\mu}_s$')
        axes.set_ylabel("Voltage (V)")
        fig.tight_layout()

    def plot_eval(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.y_new.squeeze(), label="Ground Truth", color='k')
        sns.lineplot(ax=axes, data=self.y_sim_new.squeeze(), label="Before adaptation", color='b',
                     linestyle='dashed', linewidth=0.75)
        sns.lineplot(ax=axes, data=self.y_lin_new.squeeze(), label="Linear adaptation", color='r',
                     linestyle='dashed', linewidth=0.75)

        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel=r'$\mathrm{Voltage (v_C)}$', xlabel='Samples')
        axes.legend(handles, labels, ncol=3, loc='upper center', frameon=False, fontsize=8)
        fig.tight_layout()

    def plot_inputs(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.u_torch_val.squeeze(), label="$\mathrm{v_{in}}$")
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel=r'$\mathrm{Voltage (v_{in})}$', xlabel='Samples')
        axes.legend(handles, labels, ncol=3, loc='upper center', frameon=False, fontsize=10)
        fig.tight_layout()

    def plot_outputs(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.y.squeeze(), label="$\mathrm{v_{C}}$", linewidth=0.75)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel=r'$\mathrm{Voltage (v_{C})}$', xlabel='Samples')
        axes.legend(handles, labels, ncol=3, loc='upper center', frameon=False, fontsize=10)
        fig.tight_layout()

    def plot_eval_errors(self):
        err_lin = []
        err_sim = []
        y_new = self.y_new.squeeze()
        for i in range(y_new.shape[0]):
            e = np.sqrt(np.mean((y_new[:i] - self.y_lin_new[:i]) ** 2))
            e2 = np.sqrt(np.mean((y_new[:i] - self.y_sim_new[:i]) ** 2))
            err_lin.append(e)
            err_sim.append(e2)
        fig, axes = plt.subplots(1, 1)
        axes.plot(err_lin, label="Linear adaptation")
        axes.plot(err_sim, label="Before adaptation")
        axes.set_ylabel("RMSE")
        axes.set_xlabel("Number of samples")
        axes.legend()
        plt.grid()
        fig.tight_layout()

    def plot_EKF_eval(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(ax=axes, data=self.y_new[2:].squeeze(), label="Ground Truth",
                     linewidth=0.5)
        axes.axvline(100, ls='--', color='black', linewidth=1.5, alpha=.75)
        axes.axvline(250, ls='--', color='black', linewidth=1.5, alpha=.75)
        sns.lineplot(ax=axes, data=self.y_sim_new.squeeze(), label="Nominal",
                     linestyle='dashed', linewidth=0.75)
        sns.lineplot(ax=axes, data=self.y_EKF_pred_eval.squeeze(), label="EKF",
                     linestyle='dashed', linewidth=0.75)
        sns.lineplot(ax=axes, data=self.y_lin_new.squeeze(), label="BLR",
                     linestyle='dashed', linewidth=0.75)

        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel=r'$\mathrm{v_C (V)}$', xlabel='Samples')
        axes.legend(handles, labels, ncol=4, loc='lower center', frameon=False, fontsize=8)
        axes.set_ylim([-0.03, 0.03])
        fig.tight_layout()

        # R-squared metrics
        R_sq = metrics.r_squared(self.y_EKF_pred_eval.squeeze(), self.y_EKF_true_eval[2:].squeeze())
        print(f"R-squared EKF prediction: {R_sq}")

    def plot_EKF_eval_zoom(self):
        sns.set_style("whitegrid")   # 250:550
        fig, axes = plt.subplots(1, 1)
        #sns.lineplot(ax=axes, data=self.y_EKF_true_eval.squeeze()[100:450], label="Ground Truth",
        #             linewidth=0.75)
        sns.lineplot(ax=axes, data=self.y_new.squeeze()[100:250], label="Ground Truth",
                     linewidth=1.)
        sns.lineplot(ax=axes, data=self.y_sim_new.squeeze()[100:250], label="Nominal",
                     linestyle='dashed', linewidth=1.)
        sns.lineplot(ax=axes, data=self.y_EKF_pred_eval.squeeze()[100:250], label="EKF",
                     linestyle='dashed', linewidth=1.)
        sns.lineplot(ax=axes, data=self.y_lin_new.squeeze()[100:250], label="BLR",
                     linestyle='dashed', linewidth=1.)
        axes.axvline(0, ls='--', color='black', linewidth=1.5, alpha=.75)
        axes.axvline(149, ls='--', color='black', linewidth=1.5, alpha=.75)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([], [], frameon=False)
        axes.set(ylabel=r'$\mathrm{v_C (V)}$', xlabel='Samples')
        axes.legend(handles, labels, ncol=4, loc='lower center', frameon=False, fontsize=8)
        axes.set_ylim([-0.03, 0.03])
        # axes.set_xticks(range(100, 450, 50))
        axes.set_xticklabels(range(80, 270, 20))

        # axes.set_xlim([100, 449])
        fig.tight_layout()

        # R-squared metrics
        R_sq = metrics.r_squared(self.y_EKF_pred_eval.squeeze(), self.y_EKF_true_eval[2:].squeeze())
        print(f"R-squared EKF prediction: {R_sq}")

if __name__ == "__main__":
    pt = PlotRLC()
    pt.plot_EKF_eval_zoom()
    plt.show()