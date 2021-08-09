import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import dimod




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
b = -2*np.matmul(XP.transpose(), df_transformed[:,0])


bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)


for k in range(A.shape[0]):
    bqm.set_linear('x' + str(k), b[k])


for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[0]):
        if not A[i,j] == 0:
            bqm.set_quadratic('x' + str(i), 'x' + str(j), A[i,j]) 

     
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm)
#, num_reads=5000, label='Protac Regression Test 1'
print(sampleset)

