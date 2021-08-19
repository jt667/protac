import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

df = pd.read_csv("protac cleaned.csv")


df= df.drop(["Smiles", 'Dmax (%)',
 'IC50 (nM, Protac to Target)',
 'IC50 (nM, Cellular activities)'], axis=1)
df=df.dropna(subset=["DC50 (nM)"])
df = df.drop(df[ df["Sensor"] != 0].index).drop(["Sensor"],axis=1)
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)
df["DC50 (nM)"] = np.log2(df["DC50 (nM)"])
#df["HighLow"] = df["DC50 (nM)"].ge(8.23).astype(int)
df["HighLow"] = df["DC50 (nM)"].ge(8.23).astype(int)

scaler = StandardScaler()
df_transformed = scaler.fit_transform(df.drop(["DC50 (nM)","HighLow"],axis=1))
df_transformed = np.hstack((df_transformed,np.ones((df_transformed.shape[0],1))))

X_train, X_test, y_train, y_test = train_test_split( 
    df_transformed, df["HighLow"],test_size=0.2, random_state=42)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


print(accuracy_score(predictions,y_test))
print(f1_score(predictions, y_test))