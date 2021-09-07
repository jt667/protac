import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import dimod
from azure.quantum import Workspace
from azure.quantum.optimization import Problem, ProblemType, Term, SimulatedAnnealing

# Quantum Workspace details
# These can be found in the Azure Portal in the workspace overview
workspace = Workspace (
  subscription_id = "",
  resource_group = "",
  name = "",
  location = ""
)


df = pd.read_csv("protac_cleaned.csv")
df = df.drop(df[ df["DC50 (nM)"] > 30000].index)
y = np.log2(df["DC50 (nM)"])
df = df.drop(["DC50 (nM)"],axis=1)


scaler = StandardScaler()
X = scaler.fit_transform(df)
X = np.hstack((X,np.ones([X.shape[0],1])))

P = np.array([-1, -0.5, 0.5, 1, 2, 4])
curlyP = np.kron(np.eye(12),P)
XP = np.matmul(X,curlyP)

A = np.matmul(XP.transpose(),XP)
b = -2*np.matmul(XP.transpose(), df_transformed[:,0])

# Create problem terms
terms = []
#bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)

# Add linear terms
for k in range(A.shape[0]):
    terms.append(Term(c=float(b[k]), indices=[k]))
    #bqm.set_linear('x' + str(k), b[k])

# Add quadratic terms
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[0]):
        if not A[i,j] == 0:
            terms.append(Term(c=float(A[i,j]), indices=[i,j]))
            #bqm.set_quadratic('x' + str(i), 'x' + str(j), A[i,j]) 
  
#sampler = EmbeddingComposite(DWaveSampler())
#sampleset = sampler.sample(bqm)
#, num_reads=5000, label='Protac Regression Test 1'
#print(sampleset)

# Name our problem
problem = Problem(name="Protac", problem_type=ProblemType.pubo)
problem.add_terms(terms=terms)

# Choose our solver
solver = SimulatedAnnealing(workspace, timeout=100)

# Submit the problem to the solver
result = solver.optimize(problem)
print(result)
