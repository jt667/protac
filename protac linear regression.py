import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler





df = pd.read_csv("protac cleaned.csv")


df= df.drop(["Smiles", 'Dmax (%)',
 'IC50 (nM, Protac to Target)',
 'IC50 (nM, Cellular activities)'], axis=1)
df=df.dropna(subset=["DC50 (nM)"])
df["ones"] = df.shape[0]*[1]

scaler = StandardScaler()
df_transformed = scaler.fit_transform(df)
X = np.delete(df_transformed,0,1)

P = np.array([-1, -0.5, 0.5, 1, 2, 4])
curlyP = np.kron(np.eye(12),P)
XP = np.matmul(X,curlyP)



A = np.matmul(XP.transpose(),XP)
b = -2*np.matmul(XP.transpose(), df_transformed[:,0] )

