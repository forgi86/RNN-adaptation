Main scripts:

- 00_RLC_generate_{dataset}.py: generate the datasets for:
    * train: train the neural network model
    * test: test the neural network model
    * transfer: adapt the linearized model using the nominal model's Jacobian to generate the features
    * eval: evaluate the performance of the adapted GP-like model on new data
Note: the datasets are already on the repo, no need to run the scripts above

- 01_RLC_train.py: fit a nominal state-space model on the training dataset
- 02_RLC_test.py: test the nominal state-space model performance on the test dataset
- 03_RLC_transfer: estimate the parameters of the linear model on the transfer dataset
- 04_RLC_eval: evaluate the parameters of the linear model on the eval dataset
- RLC_retrain: alternative (and basic) model adaptation: full re-train on the transfer dataset