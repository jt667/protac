import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import dimod




df = pd.read_csv("protac_cleaned.csv")
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)
y = np.log2(df["DC50 (nM)"])
df = df.drop(["DC50 (nM)"],axis=1)


scaler = StandardScaler()
X = scaler.fit_transform(df)
X = np.hstack((X,np.ones([X.shape[0],1])))



P = np.array([-1, -0.5, 0.5, 1, 2, 4])
curlyP = np.kron(np.eye(X.shape[1]),P)
XP = np.matmul(X,curlyP)



A = np.matmul(XP.transpose(),XP)
b = -2*np.matmul(XP.transpose(), y)


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

