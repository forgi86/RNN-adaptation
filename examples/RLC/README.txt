Main scripts:

- RLC_generate_*.py: generate the datasets for:
    - train: train the neural network model
    - test: test the neural network model
    - transfer: adapt the linearized (GP-like) model using the nominal model's Jacobian to generate the features
    - eval: evaluate the performance of the linearized GP-like model on new data

- RLC_train.py: fit a dynoNet model on the training dataset
- RLC_test.py: test the dynoNet model on the test dataset
- RLC_transfer_parspace_naive.py: adapt the linearized model in parameter space building the Jacobian matrix explicitly
- RLC_transfer_parspace_lazy.py: adapt the linearized model in parameter space using the Fisher-vector trick, using the LazyTensors from the paper
- RLC_transfer_gp.py: adapt the linearized model using the GPytorch and the NTK Kernel. Inference in function- or parameter- space according to use_linearstrategy param

- compare_techniques.py: compare results of different inference strategies


Usage:
- Run RLC_generate_{train, test, transfer,eval}.py (optional step, datasets are already generated)
- Run RLC_train.py to obtain the nominal model
- Run RLC_transfer_{parspace_naive, parspace_lazy, gp}.py (for the latter, run twice with use_linearstrategy = {True, False}
- Run compare techniques.py