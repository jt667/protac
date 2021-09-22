import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import circuits
import pennylane as qml
from pennylane import numpy as np
import optimisation_functions


##
## Customisable parameters
##

# The number of sets of randomly generated hyperparameters we're testing
# (any int)
hyperparameter_number = 1
# Number of mini batches
mini_batch_number = 20
# Number of epochs
epochs = 1
# To be considered, the accuracy on the training set must be on average at
# least this accuracy (value between 0 and 1)
threshold_accuracy = 0.7
# Circuit Models we're testing (takes lists of integers from 1 to 19)
circuit_numbers = [1]
# Split the data into a hyperparameter tuning / model selection / final model
# or hypeparameter tuning + model selection / final model
data_splits = [0.9]


# Which run should we perform?
jobID = "interactive"
run = 1
if len(sys.argv) > 1:
    try:
        jobID = sys.argv[1]
        run = int(sys.argv[2])
    except ValueError:
        print("Invalid run number!")
        raise
print("\nJob: {}, Run: {}\n".format(jobID, run))

# Read customisable parameters from a file instead
file_path = "run_parameters//run" + str(run) +".json"


if len(file_path) > 0:
    cp = optimisation_functions.customisable_parameters(file_path=file_path, run=run)
else:      
    cp = optimisation_functions.customisable_parameters(
        hyperparam_num=hyperparameter_number,
        mini_batch_num=mini_batch_number,
        epochs=epochs,
        threshold_acc=threshold_accuracy,
        circuit_nums=circuit_numbers,
        data_splits = data_splits
    )


##
## Prepare the data
##

x,y = optimisation_functions.data_prep(splits=cp.data_splits)
skf = StratifiedKFold()

if len(cp.data_splits) == 2:
    print("Hyperparameter Tuning / Model Selection / Final Model - Data Split")
    x_train, x_test_select, x_test_final = x
    y_train, y_test_select, y_test_final = y
else:
    print("Hyperparameter Tuning + Model Selection / Final Model - Data Split")
    x_train, x_test_final = x
    y_train, y_test_final = y



##
## Choose best set of hyper parameters by cross validation
##

circuit_scores = []
np.random.seed(15)

for circ_num in cp.circuit_nums:
    # Reset best f1 score tracker
    best_score = 0

    # Number of training parameters for circuit + linear model
    parameter_num = circuits.parameter_count(circ_num, x_train.shape[1]) + x_train.shape[1] + 1

    # Generate random values for training parameters and
    # stepsize, beta1, beta2 values of an Adam Optimiser
    var_starter = np.random.randn(cp.hyperparam_num, parameter_num)
    steps = np.random.beta(2, 20, cp.hyperparam_num)
    beta1_range = 0.01 * np.random.standard_normal(cp.hyperparam_num) + 0.9
    beta2_range = 0.001 * np.random.beta(2, 2, cp.hyperparam_num) + 0.9985

    # Collect hyperparameters together
    hyperparam_collection = zip(var_starter, steps, beta1_range, beta2_range)

    for v, s, b1, b2 in hyperparam_collection:
        # Reset f1_scores and accuracy score trackers
        f1_scores = []
        acc_scores = []

        # Initisalise parameters and optimiser
        var_init = v
        opt = qml.AdamOptimizer(stepsize=s, beta1=b1, beta2=b2)

        # 5 Fold cross validation over 60% stratified sample of full dataset using best mean f1 score to
        # tune hyperparameters
        for train_index, test_index in skf.split(x_train, y_train):
            # Generate folds for testing and reset optimiser
            x_fold = x_train[train_index]
            y_fold = y_train[train_index]
            var = var_init
            opt.reset()

            batch_size = int(x_fold.shape[0] / cp.mini_batch_num)
            for ep in range(cp.epochs):
                for it in range(cp.mini_batch_num):
                    # Update the weights by one optimizer step

                    x_train_batch = x_fold[it * batch_size : (it + 1) * batch_size]
                    y_train_batch = y_fold[it * batch_size : (it + 1) * batch_size]
                    var = opt.step(
                        lambda v: optimisation_functions.cost(
                            v, x_train_batch, y_train_batch, circ_num
                        ),
                        var,
                    )
            # Calculate f1 and accuracy scores
            pred_vals = [
                round(optimisation_functions.classifier(var, f, circ_num))
                for f in x_train[test_index]
            ]
            f1_scores.append(f1_score(y_train[test_index], pred_vals))
            acc_scores.append(accuracy_score(y_train[test_index], pred_vals))
        # Save the first set of parameters
        if s == steps[0]:
            best_score = np.mean(f1_scores)
            hyperparams = optimisation_functions.hyperparameters(
                circuit_num=circ_num,
                epochs=epochs,
                starting_var=var_init,
                step_size=s,
                beta1=b1,
                beta2=b2,
            )
        # Check if current f1score is highest and model meets the accuracy threshold.
        # Store it if so.
        elif np.mean(f1_scores) > best_score and np.mean(acc_scores) > cp.threshold_acc:
            best_score = np.mean(f1_scores)
            hyperparams = optimisation_functions.hyperparameters(
                starting_var=var_init,
                circuit_num=circ_num,
                step_size=s,
                beta1=b1,
                beta2=b2,
            )
    # Save the best hyperparameters locally and in a .txt file
    circuit_scores.append([best_score, hyperparams])
    # hyperparams.save_params_json()

    print("Completed Circuit " + str(circ_num))
print("Hyperparameter tuning completed")    

##
## Select best circuit model
##

test_scores = []
circuit_info = []

if len(cp.data_splits) == 2:
    for i in range(len(cp.circuit_nums)):
        hyperparams = circuit_scores[i][1]
    
        var = hyperparams.starting_var
        opt = qml.AdamOptimizer(
            stepsize=hyperparams.step_size, beta1=hyperparams.beta1, beta2=hyperparams.beta2
        )
        batch_size = int(x_train.shape[0] / cp.mini_batch_num)
        for ep in range(cp.epochs):
            for it in range(cp.mini_batch_num):
                # Update the weights by one optimizer step
    
                x_train_batch = x_train[it * batch_size : (it + 1) * batch_size]
                y_train_batch = y_train[it * batch_size : (it + 1) * batch_size]
                var = opt.step(
                    lambda v: optimisation_functions.cost(
                        v, x_train_batch, y_train_batch, cp.circuit_nums[i]
                    ),
                    var,
                )
        pred_vals = [
            round(optimisation_functions.classifier(var, f, hyperparams.circuit_num))
            for f in x_test_select
        ]
        f1Score = f1_score(y_test_select, pred_vals)
        acc = accuracy_score(y_test_select, pred_vals)
    
        test_scores.append(f1Score)
        circuit_info.append([f1Score, acc])
    
        print("Completed Circuit " + str(hyperparams.circuit_num))
        print(circuit_info[i])
    print("Model Selection Completed")
    print(np.max(test_scores))
    print(cp.circuit_nums[np.argmax(test_scores)])
    

##
## Final scoring
##

# Determine the model with best f1 score on the test selection data
if len(cp.data_splits) == 1:
    test_scores = [model[0] for model in circuit_scores]
    
param_index = np.argmax(test_scores)
model_num = cp.circuit_nums[param_index]
hyperparams = circuit_scores[param_index][1]
    



var = hyperparams.starting_var
opt = qml.AdamOptimizer(
    stepsize=hyperparams.step_size, beta1=hyperparams.beta1, beta2=hyperparams.beta2
)

batch_size = int(x_train.shape[0] / cp.mini_batch_num)
for ep in range(cp.epochs):
    for it in range(cp.mini_batch_num):
        # Update the weights by one optimizer step

        x_train_batch = x_train[it * batch_size : (it + 1) * batch_size]
        y_train_batch = y_train[it * batch_size : (it + 1) * batch_size]
        var = opt.step(
            lambda v: optimisation_functions.cost(
                v, x_train_batch, y_train_batch, model_num
            ),
            var,
        )
pred_vals = [
    round(optimisation_functions.classifier(var, f, model_num)) for f in x_test_final
]
final_score = f1_score(y_test_final, pred_vals)
final_acc = accuracy_score(y_test_final, pred_vals)
print(final_score)
print(final_acc)

# Save the hyper parameters
hyperparams.save_final_vars(var)
#hyperparams.save_params_json(
#    "hyperparameters//602020best_params" + str(cp.hyperparam_num) + "_trials.json"
#)
