from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import optimisation_functions

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


# File containing hyperparameters
param_path = "hyperparameters//602020best_params1_trials.txt"


# Load model
hyperparams = optimisation_functions.hyperparameters()
hyperparams.load_params(param_path)

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
