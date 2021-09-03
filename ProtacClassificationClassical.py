import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score


## Preparation
##

df = pd.read_csv("protacCleaned.csv")

# Drop the biggest outlier 
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)

# Move to log scale on concentration
df["DC50 (nM)"] = np.log2(df["DC50 (nM)"])

# Classify upper 25% quartile as 1, rest as 0
df["HighLow"] = df["DC50 (nM)"].ge(8.23).astype(int)

# Rescale all data
scaler = StandardScaler()
dfTransformed = scaler.fit_transform(df.drop(["DC50 (nM)","HighLow"],axis=1))

# Split data into 80/20 training / test
X_train, X_test, y_train, y_test = train_test_split( 
    dfTransformed, df["HighLow"],test_size=0.2, random_state=42)


## Model Training
##


# Train a simple random forest classifier on traiing set
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


## Final Predictions

# Compute predictions on test set, accuracy and f1 score
predictions = clf.predict(X_test)
print(accuracy_score(predictions,y_test))
print(f1_score(predictions, y_test))
