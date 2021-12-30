Main scripts:

- 00_cstr_generate.py: generate the datasets for:
    * train: train the neural network model
    * test: test the neural network model
    * transfer: adapt the linearized model using the nominal model's Jacobian to generate the features
    * eval: evaluate the performance of the adapted GP-like model on new data

- 01_cstr_train.py: fit a nominal LSTM model on the training dataset (lstm.pt)
- 02_cstr_test.py: test the nominal LSTM model performance on the test dataset
- 03_cstr_transfer: estimate the parameters of the linear model on the transfer dataset (theta_lin_cf)
- 04_cstr_eval: evaluate the parameters of the linear model on the eval dataset
- cstr_retrain: alternative (and basic) model adaptation: full re-train on the transfer dataset