from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import optimisation_functions
import pennylane as qml

##
## Prepare the data and split into sets 60/20/20
##

x, y = optimisation_functions.dataPrep()
x_train, x_test_select, y_train, y_test_select = train_test_split(
    x, y, test_size=0.4, random_state=15
)
x_test_select, x_test_final, y_test_select, y_test_final = train_test_split(
    x_test_select, y_test_select, test_size=0.5, random_state=15
)


# Number of mini batches
mini_batch_num = 20


# File containing hyperparameters
param_path = "hyperparameters//Circuit2Params.txt"


# Load model
hyperparams = optimisation_functions.hyperparameters()
hyperparams.load_params(param_path)

# If the model has not been fully trained - train it
if not hasattr(hyperparams, "final_vars"):
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
                v, x_train_batch, y_train_batch, hyperparams.circuit_num
            ),
            var,
        )
    hyperparams.save_final_vars(var)
# Compute predictions and scores
pred_vals = [
    round(
        optimisation_functions.classifier(
            hyperparams.final_vars, f, hyperparams.circuit_num
        )
    )
    for f in x_test_final
]
final_score = f1_score(y_test_final, pred_vals)
final_acc = accuracy_score(y_test_final, pred_vals)
print(final_score)
print(final_acc)
