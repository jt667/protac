import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import klib
import pennylane as qml
from pennylane import numpy as np


def square_loss(labels, predictions):
    # Compute the squared loss function
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def log_loss(labels, predictions):
    # Compute the log loss function
    loss = 0
    for l, p in zip(labels, predictions):
        loss += l * np.log(p) + (1 - l) * np.log(1 - p)
    return (-1) * loss / len(labels)


def cost(var, X, Y):
    # Compute class predictions
    predictions = [classifier(var, x) for x in X]
    # return square_loss(Y, predictions)
    return log_loss(Y, predictions)


def classifier(var, x):
    dev = qml.device("default.qubit", wires=x.size)
    circuit = qml.QNode(circuit15, dev)
    measurements = circuit(var, x)

    lin_model = np.inner(measurements, var[-11:-1]) + var[-1]
    sig = 1 / (1 + np.exp((-1) * lin_model))

    return sig


def scale(data):
    for i in range(data.shape[1]):
        col_min = np.min(data[:, i])
        col_max = np.max(data[:, i])
        data[:, i] = 2 * np.pi * (data[:, i] - col_min) / (col_max - col_min) - np.pi
    return data


def circuit15(params, x):
    n = x.size
    # Requires 2n parameters
    qml.templates.AngleEmbedding(x, wires=[i for i in range(n)])
    index = 0
    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1
    # CNOT ring
    qml.CNOT(wires=[n - 1, 0])
    index += 1

    for i in range(n - 1):
        qml.CNOT(wires=[n - 2 - i, n - 1 - i])
        index += 1
    for i in range(n):
        # RY block
        qml.RY(params[index], wires=i)
        index += 1
    # CNOT ring
    qml.CNOT(wires=[n - 1, n - 2])
    index += 1
    qml.CNOT(wires=[0, n - 1])
    index += 1

    for i in range(n - 2):
        qml.CNOT(wires=[i + 1, i])
        index += 1
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return measurements


# Prepares the data for classification
df = pd.read_csv("protac_cleaned.csv")
# Drop the outlier
df = df.drop(df[df["DC50 (nM)"] > 30000].index)
# Create classes for high and low concentations of DC50
df["HighLow"] = df["DC50 (nM)"].ge(301).astype(int)

# Rescale into [-pi,pi]
scaler = StandardScaler()
df_transformed = scaler.fit_transform(df.drop(["DC50 (nM)", "HighLow"], axis=1))
df_transformed = scale(df_transformed)


X_train, X_test, y_train, y_test = train_test_split(
    df_transformed, df["HighLow"], test_size=0.2, random_state=42
)


opt = qml.AdamOptimizer()

np.random.seed(32)
var_init = 0.1 * np.random.randn(141)

batch_size = 20

var = var_init
for it in range(19):
    # Update the weights by one optimizer step

    X_train_batch = X_train[it * batch_size : (it + 1) * batch_size]
    y_train_batch = y_train[it * batch_size : (it + 1) * batch_size]
    var = opt.step(lambda v: cost(v, X_train_batch, y_train_batch), var)

    # Compute predictions on train and validation set
    predictions_train = [round(classifier(var, f)) for f in X_train]
    predictions_val = [round(classifier(var, f)) for f in X_test]

    # # Compute accuracy on train and validation set
    acc_train = accuracy_score(y_train, predictions_train)
    acc_val = accuracy_score(y_test, predictions_val)
    f1_train = f1_score(y_train, predictions_train)
    f1_val = f1_score(y_test, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} \n | f1 train: {:0.7f} | f1 validation: {:0.7f}"
        "".format(
            it + 1,
            cost(var, df_transformed, df["HighLow"]),
            acc_train,
            acc_val,
            f1_train,
            f1_val,
        )
    )
    print("")
print("Training complete")


predictions_val = [round(classifier(var, f)) for f in X_test]
acc_val = accuracy_score(y_test, predictions_val)
print(acc_val)
print(f1_score(y_test, predictions_val))
