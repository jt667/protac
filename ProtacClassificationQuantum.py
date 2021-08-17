import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import klib
import Circuits
import pennylane as qml
from pennylane import numpy as np


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def log_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += l*np.log(p) + (1-l)*np.log(1-p) 
    
    return (-1)*loss/len(labels)

def cost(var, X, Y):
    predictions = [classifier(var, x) for x in X]
    return square_loss(Y, predictions)
    #return log_loss(Y,predictions)


def classifier(var,x):
    #dev = qml.device("qiskit.ibmq",wires=x.size, backend='ibmq_qasm_simulator', shots=100)
    dev = qml.device("default.qubit", wires=x.size)
    circuit = qml.QNode(Circuits.circuit5,dev)
    measurements = circuit(var,x)
    
    lin_model = np.inner(measurements,var[-11:-1]) + var[-1] 
    sig = 1 / (1 + np.exp((-1)*lin_model))

    return sig


def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


def scale(data):
    for i in range(data.shape[1]):  
        col_min =np.min(data[:,i])
        col_max =np.max(data[:,i])
        data[:,i] = 2*np.pi * (data[:,i] - col_min)/(col_max -col_min) - np.pi
    return data
        

df = pd.read_csv("protacCleaned.csv")
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)
df["DC50 (nM)"] = np.log2(df["DC50 (nM)"])
df["HighLow"] = df["DC50 (nM)"].ge(8.23).astype(int)



scaler = StandardScaler()
df_transformed = scaler.fit_transform(df.drop(["DC50 (nM)","HighLow"],axis=1))
df_transformed = scale(df_transformed)



X_train, X_test, y_train, y_test = train_test_split( 
    df_transformed, df["HighLow"],test_size=0.2,random_state=42)



opt = qml.RMSPropOptimizer(stepsize=0.1)

np.random.seed(32)
var_init = (0.1*np.random.randn(141))

batch_size = 20

var= var_init
for it in range(19):
    # Update the weights by one optimizer step

    X_train_batch = X_train[it*batch_size:(it +1)*batch_size]
    y_train_batch = y_train[it*batch_size:(it +1)*batch_size]
    var = opt.step(lambda v: cost(v, X_train_batch, y_train_batch), var)

 
    # Compute predictions on train and validation set
    # predictions_train = [round(classifier(var, f)) for f in X_train]
    # predictions_val = [round(classifier(var, f)) for f in X_test]

    # # Compute accuracy on train and validation set
    # acc_train = accuracy(y_train, predictions_train)
    # acc_val = accuracy(y_test, predictions_val)
    
    # print(
    #     "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
    #     "".format(it + 1, cost(var, df_transformed, df["HighLow"]), acc_train, acc_val)
    # )

print("Training complete")
    
   
predictions_val = [round(classifier(var, f)) for f in X_test]
acc_val = accuracy(y_test, predictions_val)
print(acc_val)
print(f1_score(y_test, predictions_val))
print(sum(predictions_val))
