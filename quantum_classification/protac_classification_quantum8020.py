from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import circuits
import pennylane as qml
from pennylane import numpy as np
import optimisation_functions

##
## Prepare the data and split into sets 80/20
##

x, y = optimisation_functions.dataPrep()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=15
)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
skf = StratifiedKFold()

##
## Customisable parameters
##

# The number of sets of randomly generated hyperparameters we're testing
# (any int)
hyperparam_num = 40
# Number of mini batches
mini_batch_num = 20
# To be considered, the accuracy on the training set must be on average at
# least this accuracy (value between 0 and 1)
threshold_acc = 0.7
# Circuit Models we're testing (takes lists of integers from 1 to 19)
circuit_nums = [1, 2, 5, 10, 12, 13, 16, 18]

##
## Choose best set of hyper parameters by cross validation
##

circuit_scores = []
np.random.seed(15)

for circ_num in circuit_nums:
    # Reset best f1 score tracker
    best_score = 0

    # Number of training parameters for circuit + linear model
    parameter_num = circuits.parameter_count(circ_num, x.shape[1]) + x.shape[1] + 1

    # Generate random values for training parameters and
    # stepsize, beta1, beta2 values of an Adam Optimiser
    var_starter = np.random.randn(hyperparam_num, parameter_num)
    steps = np.random.beta(2, 20, hyperparam_num)
    beta1_range = 0.01 * np.random.standard_normal(hyperparam_num) + 0.9
    beta2_range = 0.001 * np.random.beta(2, 2, hyperparam_num) + 0.9985

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

            batch_size = int(x_fold.shape[0] / mini_batch_num)

            for it in range(mini_batch_num):
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
                starting_var=var_init,
                circuit_num=circ_num,
                step_size=s,
                beta1=b1,
                beta2=b2,
            )
        # Check if current f1score is highest and model meets the accuracy threshold.
        # Store it if so.
        elif np.mean(f1_scores) > best_score and np.mean(acc_scores) > threshold_acc:
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
    hyperparams.save_params()

    print("Completed Circuit " + str(circ_num))
##
## Select best circuit model
##

test_scores = []
circuit_info = []

for i in range(len(circuit_nums)):
    hyperparams = circuit_scores[i][1]

    var = hyperparams.starting_var
    opt = qml.AdamOptimizer(
        stepsize=hyperparams.step_size, beta1=hyperparams.beta1, beta2=hyperparams.beta2
    )
    batch_size = int(x_train.shape[0] / mini_batch_num)

    for it in range(mini_batch_num):
        # Update the weights by one optimizer step

        x_train_batch = x_train[it * batch_size : (it + 1) * batch_size]
        y_train_batch = y_train[it * batch_size : (it + 1) * batch_size]
        var = opt.step(
            lambda v: optimisation_functions.cost(
                v, x_train_batch, y_train_batch, circuit_nums[i]
            ),
            var,
        )
    pred_vals = [
        round(optimisation_functions.classifier(var, f, hyperparams.circuit_num))
        for f in x_test
    ]
    f1Score = f1_score(y_test, pred_vals)
    acc = accuracy_score(y_test, pred_vals)

    test_scores.append(f1Score)
    circuit_info.append([f1Score, acc])

    print("Completed Circuit " + str(hyperparams.circuit_num))
    print(circuit_info[i])
print(np.max(test_scores))
print(circuit_nums[np.argmax(test_scores)])

##
## Final scoring
##

# Determine the model with best f1 score on the test selection data
param_index = np.argmax(test_scores)
model_num = circuit_nums[param_index]
hyperparams = circuit_scores[param_index][1]

# Save the hyper parameters
hyperparams.save_params(
    "hyperparameters//8020bestParams" + str(hyperparam_num) + "trials.txt"
)

var = hyperparams.starting_var
opt = qml.AdamOptimizer(
    stepsize=hyperparams.step_size, beta1=hyperparams.beta1, beta2=hyperparams.beta2
)

batch_size = int(x_train.shape[0] / mini_batch_num)

for it in range(mini_batch_num):
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
    round(optimisation_functions.classifier(var, f, model_num)) for f in x_test
]
final_score = f1_score(y_test, pred_vals)
final_acc = accuracy_score(y_test, pred_vals)
print(final_score)
print(final_acc)
