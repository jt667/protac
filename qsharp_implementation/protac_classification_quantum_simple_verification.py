from sklearn.metrics import f1_score, accuracy_score
import optimisation_functions
import numpy as np


def classifier(var,x):
    measurements = circuit15(var, x)

# Put outputs into a linear model with sigmoid activation function
    lin_model = np.inner(measurements, var[-11:-1]) + var[-1]
    sig = 1 / (1 + np.exp((-1) * lin_model))
    
    return sig



##
## Prepare the data and split into sets 60/20/20
##

# File containing hyperparameters
param_path = "qsharp_test_circuit_15.json"
verification_model = optimisation_functions.load_params_json(param_path)

x,y = optimisation_functions.data_prep(splits=[0.6,0.2])
x_train, x_test_select, x_test_final = x
y_train, y_test_select, y_test_final = y



# Compute predictions and scores
pred_vals = [
    round(
        classifier(
            verification_model.final_vars, f
            )
    )
    for f in x_test_final
]

final_score = f1_score(y_test_final, pred_vals)
final_acc = accuracy_score(y_test_final, pred_vals)
print("f1 Score: " + str(final_score))
print("Accuracy: " + str(final_acc))
