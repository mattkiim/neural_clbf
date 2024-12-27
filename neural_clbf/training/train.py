import os

# Define hyperparameter ranges
hidden_sizes = [32, 64, 128]
lambda_values = [0.1, 1.0, 10.0]
relaxation_penalties = [50.0, 100.0, 500.0]

# Iterate over hyperparameter combinations
for hidden_size in hidden_sizes:
    for lambda_value in lambda_values:
        for penalty in relaxation_penalties:
            print(f"Running with clbf_hidden_size={hidden_size}, clf_lambda={lambda_value}, clf_relaxation_penalty={penalty}")
            os.system(f"python train_mvc_rel.py --clbf_hidden_size {hidden_size} --clf_lambda {lambda_value} --clf_relaxation_penalty {penalty}")
