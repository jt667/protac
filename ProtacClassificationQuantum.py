import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import klib
import Circuits
import pennylane as qml



def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cost(var, X, Y):
    predictions = [classifier(var, x) for x in X]
    return square_loss(Y, predictions)


def classifier(var,x):
    
    measurements = Circuits.circuit19(var,11,x)
    neg_count = len(list(filter(lambda x: (x <0), measurements)))
    if neg_count > len(measurements)/2:
        return 0
    else:
       return 1

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


df = pd.read_csv("protac cleaned.csv")


df= df.drop(["Smiles", 'Dmax (%)',
 'IC50 (nM, Protac to Target)',
 'IC50 (nM, Cellular activities)'], axis=1)
df=df.dropna(subset=["DC50 (nM)"])
df = df.drop(df[ df["Sensor"] != 0].index).drop(["Sensor"],axis=1)
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)
df["DC50 (nM)"] = np.log2(df["DC50 (nM)"])
df["HighLow"] = df["DC50 (nM)"].ge(8.23).astype(int)

scaler = StandardScaler()
df_transformed = scaler.fit_transform(df.drop(["DC50 (nM)","HighLow"],axis=1))
df_transformed = np.hstack((df_transformed,np.ones((df_transformed.shape[0],1))))

X_train, X_test, y_train, y_test = train_test_split( 
    df_transformed, df["HighLow"],test_size=0.2, random_state=42)



opt = qml.QNGOptimizer(0.01)

var_init = (0.01*np.random.randn(33))
num_train = int(0.8 * df_transformed.shape[0])
batch_size = 20

var= var_init
for it in range(60):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_train_batch = X_train[batch_index]
    y_train_batch = y_train[batch_index]
    var = opt.step(lambda v: cost(v, X_train_batch, y_train_batch), var)

    # Compute predictions on train and validation set
    predictions_train = [classifier(var, f) for f in X_train]
    predictions_val = [classifier(var, f) for f in X_test]

    # Compute accuracy on train and validation set
    acc_train = accuracy(y_train, predictions_train)
    acc_val = accuracy(y_test, predictions_val)
    
    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost(var, df_transformed, df["HighLow"]), acc_train, acc_val)
    )
