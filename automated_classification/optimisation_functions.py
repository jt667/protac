import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import circuits
import os
import json


def scale(data):
    # Rescales the data into the interval [-pi, pi]
    for i in range(data.shape[1]):
        col_min = np.min(data[:, i])
        col_max = np.max(data[:, i])
        data[:, i] = 2 * np.pi * (data[:, i] - col_min) / (col_max - col_min) - np.pi
    return data


def data_prep(upperBound=301,splits=[0.6,0.2], random_state=15):
    # Prepares the data for classification
    df = pd.read_csv("protac_cleaned.csv")
    # Drop the outlier
    df = df.drop(df[df["DC50 (nM)"] > 30000].index)
    # Create classes for high and low concentations of DC50
    df["HighLow"] = df["DC50 (nM)"].ge(upperBound).astype(int)

    # Rescale into [-pi,pi]
    scaler = StandardScaler()
    df_transformed = scaler.fit_transform(df.drop(["DC50 (nM)", "HighLow"], axis=1))
    df_transformed = scale(df_transformed)
    
    split_percentage1 = 1-splits[0]
    x_train, x_test_select, y_train, y_test_select = train_test_split(
        df_transformed, df["HighLow"], test_size=split_percentage1, random_state=random_state
    )
    
    if len(splits) == 2:
        split_percentage2 = splits[1]/split_percentage1
        x_test_select, x_test_final, y_test_select, y_test_final = train_test_split(
        x_test_select, y_test_select, test_size=split_percentage2, random_state=random_state
    )
        x = [x_train, x_test_select, x_test_final]
        y = [y_train.to_numpy(), y_test_select.to_numpy(), y_test_final.to_numpy()]
    else:
        x = [x_train, x_test_select]
        y = [y_train.to_numpy(), y_test_select.to_numpy()]
        
    
    return x,y


def cost(var, X, Y, circuit_num=5):
    # Compute class predictions
    predictions = [classifier(var, x, circuit_num) for x in X]
    # return square_loss(Y, predictions)
    return log_loss(Y, predictions)


def classifier(var, x, circuit_num=5):
    # Pick device circuit is executed on
    # dev = qml.device("qiskit.ibmq",wires=x.size, backend='ibmq_qasm_simulator', shots=100)
    dev = qml.device("default.qubit", wires=x.size)
    circuit = qml.QNode(getattr(circuits, "circuit" + str(circuit_num)), dev)

    # Run the circuit
    measurements = circuit(var, x)

    # Put outputs into a linear model with sigmoid activation function
    lin_model = np.inner(measurements, var[-11:-1]) + var[-1]
    sig = 1 / (1 + np.exp((-1) * lin_model))

    return sig


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


class hyperparameters:
    def __init__(
        self,
        circuit_num=1,
        epochs=1,
        starting_var=[],
        step_size=0.01,
        beta1=0.9,
        beta2=0.99,
        filepath="",
    ):
        if len(filepath) > 0:
            if filepath[-4:] == "json":
                self.load_params_json(filepath)
            else:
                self.load_params_txt(filepath)
        else:
            self.circuit_num = circuit_num
            self.epochs = epochs
            self.starting_var = starting_var
            self.step_size = step_size
            self.beta1 = beta1
            self.beta2 = beta2
    
    def arrays_to_lists(self):
        for key in vars(self):
            if isinstance(getattr(self,key),(np.ndarray)):
                setattr(self,key,getattr(self, key).tolist())
            
    def lists_to_arrays(self):
        for key in vars(self):
            if isinstance(getattr(self,key),(list)):
                setattr(self,key,np.array(getattr(self, key)))
    
    def load_params_json(self, file_path):
        with open(file_path, "r") as f:
            params_dict = json.load(f)
        for key in params_dict:
            setattr(self, key, params_dict[key])

    def load_params_txt(self, file_path):
        # Loads a set of hyperparameters from the given file path
        # Must be in the same format as the class saves in
        f = open(file_path, "r")
        params = [line.strip() for line in f.readlines()]
        f.close()
        self.circuit_num = int(params[0].split()[1])
        param_vars = params[2:-3]
        if "Final" in param_vars[int(0.5 * (len(param_vars) - 1))]:
            param_vars.remove("Final Variables:")
            param_vars = [float(x) for x in param_vars]
            n = int(0.5 * len(param_vars))
            self.starting_var = np.array(param_vars[:n])
            self.final_vars = np.array(param_vars[n:])
        else:
            self.starting_var = np.array([float(x) for x in param_vars])
        self.step_size = float(params[-3].split()[1])
        self.beta1 = float(params[-2].split()[1])
        self.beta2 = float(params[-1].split()[1])

    def save_final_vars(self, var):
        self.final_vars = var

    def save_params_json(self, file_path=""):
        self.arrays_to_lists()
        if not os.path.exists("hyperparameters"):
            os.mkdir("hyperparameters")
        if len(file_path) == 0:
            f = open(
                "hyperparameters//Circuit" + str(self.circuit_num) + "Params.json", "w"
            )
        else:
            f = open(file_path, "w")
        json.dump(vars(self), f)
        f.close()
        self.lists_to_arrays()

    def save_params_txt(self, file_path=""):
        # Save all parameters as .txt file in current working directory
        if not os.path.exists("hyperparameters"):
            os.mkdir("hyperparameters")
        if len(file_path) == 0:
            f = open(
                "hyperparameters//Circuit" + str(self.circuit_num) + "Params.txt", "w"
            )
        else:
            f = open(file_path, "w")
        f.write("CircuitNum: " + str(self.circuit_num) + "\n")
        f.write("Starting Variables: \n")
        for i in self.starting_var:
            f.write(str(i) + "\n")
        if hasattr(self, "final_vars"):
            f.write("Final Variables: \n")
            for i in self.final_vars:
                f.write(str(i) + "\n")
        f.write("Stepsize: " + str(self.step_size) + "\n")
        f.write("Beta1: " + str(self.beta1) + "\n")
        f.write("Beta2: " + str(self.beta2))
        f.close()


class customisable_parameters:
    def __init__(
        self,
        hyperparam_num=40,
        mini_batch_num=20,
        epochs=1,
        threshold_acc=0.7,
        circuit_nums=[1, 2, 5, 10, 12, 13, 16, 18],
        data_splits = [0.6,0.2],
        file_path="",
        run=1        
    ):
        if isinstance(run,int):
            run = str(run)
        if len(file_path) > 0:
            self.load_params_json(file_path, run)
        else:
            self.hyperparam_num = hyperparam_num
            self.mini_batch_num = mini_batch_num
            self.epochs = epochs
            self.threshold_acc = threshold_acc
            self.circuit_nums = circuit_nums
            self.data_splits = data_splits

    def load_params_json(self, file_path, run):
        with open(file_path, "r") as f:
            run_dict = json.load(f)
        for key in run_dict[run]:
            setattr(self, key, run_dict[run][key])
