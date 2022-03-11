For the non-linear RLC we adapted the System Identification setting from \citet{forgione_si_2021} --- where the goal is to approximate the unknown dynamical system with a neural network --- to the few-shot learning context. 

We define a task as one parameter instantiation of the dynamical system and sample resistance $R$ (unit $\Omega$), inductance $L_0$ (unit $\mathrm{\mu H}$) and capacitance $C$ (unit $\mathrm{nF}$) uniformly from distributions with ranges $[1, 14)$, $[20, 140)$ and $[100, 800)$, respectively. 
In this way we generate 512 training tasks and 256 test tasks, where each task consists of multiple sequences containing 2000 time steps discretized with $T_s = 1 \mathrm{\mu s}$. 
For each sequence the system input is filtered white noise with bandwidth $80\, \mathrm{kHz}$ and standard deviation $80 \mathrm{V}$. Similar to \citet{forgione_si_2022} we add Gaussian noise with standard deviation of 0.1 to the outputs to represent the measurement noise.
We parameterize the continuous dynamics of the system by a single layer feed-forward neural network with three input units, a hidden layer with 50 units followed by a tanh nonlinearity, and two output units corresponding to the state variables, which is also analog to \citet{forgione_si_2022}.
As \citet{forgione_si_2021} we discretize the model with the forward-Euler method using the same sampling time as during data generation. In contrast to them, we did not use the proposed consistency regularization as it is irrelevant for our experiments (see Appendix).

SubGD is setup in the following way:
We first pretrain our model on three sequences of the nominal system dynamics with $R = 3 \, \Omega$, $C = 270 \, \mathrm{nF}$, $L_0 = 50 \, \mathrm{\mu H}$ for 30 epochs, where 16 sequences of length 256 are comprised in a batch. 
Starting with this model we finetune on all training tasks and determine the preconditioning matrix and use the identified subspace to set the learning rate and number of update steps for final evaluation. 
At last, the performance is evaluated in few-shot setting on the test tasks. For that, we adapt the model with our SubGD update rule to each task on the support set defined as sequences of limited length and assess the performance on the query set, which is a another full sequence. We use the Adam optimizer \citep{kingma_2015_adam} for all training procedures. 

(1)
We compare our method to finetuning without our preconditioning matrix (Adam), MAML, Reptile and Jacobian Feature Regression (JFR).
JFR is a fast adaption technique for neural networks based on first order Taylor expansion around the pretrained model. By computing the Jacobian of this linearization on the test task one can use the resulting Bayesian linear model for inference on these tasks (\citet{forgione_si_2022, maddox_linearized_2021}).


In contrast to \citet{forgione_si_2022}, our evaluation does not focus on adaption time. Instead we analyze the sample efficiency of SubGD compared to the other methods. Figure~\ref{fig:rlc-support-vs-mse} and Table~\ref{tab:rlc-mse} show that SubGD performs particurlarly well on small support sizes. For larger support sizes Adam finetuning is almost on par. We also find that JFR adaption of the neural network is only beneficial for larger support sets, even though it does not reach the performance of SubGD. The reason could be that JFR only uses information from pretraining and the test task it is evaluated on, while SubGD can make adaptions to the model that are informed by all training tasks via the preconditioning matrix.


Related work.

The problem of adapting a neural network's parameters according to changing parameters of a dynamical system was recently addressed by \citet{forgione_si_2022}.
Instead of finetuning, they use linearizeation of the pretrained model to quickly adapt to new tasks \citet{maddox}.
In the present work, we extend this problem setting by adding the constraint of scarce data, which yields a considerably more realistic scenario. We compare the method proposed in \citet{forgione_si_2022} to SubGD and scrutinize their sample efficiency.

The problem of adapting a neural network's parameters according to changing parameters of a dynamical system was recently addressed by \citet{forgione_si_2022}.
In the present work, we extend this problem setting by adding the constraint of scarce data, which yields a considerably more realistic scenario. We compare the method proposed in \citet{forgione_si_2022}, which uses a linearization for quick adaption instead of finetuing \citet{maddox} to SubGD and scrutinize their sample efficiency (\textcolor{green}{see REF Max}).

Their method linearize the pretrained neural network and use this linearization for quick adaption to a new target task \citet{maddox}.


old versions:
(1)
We compare our method to finetuning without our preconditioning matrix (Adam) and Jacobian Feature Regression (JFR).
JFR is a fast adaption technique for neural networks based on first order Taylor expansion around the pretrained model (\citet{forgione_si_2022, maddox_linearized_2021}). This linearization results in a Bayesian linear model with the neural network Jacobian matrix as its features, which is used to make predictions on test tasks. 


## Appendix

### RLC system



### Training procedure and consistency loss

- \citet{forgione 2021} propose 'truncated simulation error minimization' 
- goal is twofold: 
    - learn a neural state space model which parameterizes the true system dynamics given as ordinary differential equation 
    - learn the 'simulated state' evolution of the neural state space model
- simulated state is the solution to the initial value problem defined by the neural state space model, i.e. the 
- 

- processes batches of subequences of the dataset containing the simualated or measured system dynamics

- learns the solution to the initial value problem

- 


