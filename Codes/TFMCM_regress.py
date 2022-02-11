import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import cvxpy as cp
import time
import mosek
from sklearn.neighbors import DistanceMetric

class LapMCM_regress(object):
    def __init__(self,opt):
        self.opt=opt

    def fit(self,X,Y,X_u):
        
        # X: Labelled points
        # Y: Labels
        # X_u: Unlabelled points
        
        # Construct graph
        self.X=np.vstack([X,X_u]) # All points labelled + unlabelled
        Y=Y.reshape(-1,1) # matrix of labels (lx1)

        if self.opt['neighbor_mode']=='connectivity':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='connectivity',include_self=False)
            W = (((W + W.T) > 0) * 1)
        elif self.opt['neighbor_mode']=='distance':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='distance',include_self=True)
            W = W.maximum(W.T)
            W = sparse.csr_matrix((np.exp(-W.data**2/4/self.opt['t']),
                                   W.indices,W.indptr),shape=(self.X.shape[0],self.X.shape[0]))
        else:
            raise Exception()

        # Delta Matrix - Graph Difference Operator
        delta = GDO(W)

        # Computing K with k(i,j) = kernel(i, j)
        K = self.opt['kernel_function'](self.X,self.X,**self.opt['kernel_parameters'])

        l=X.shape[0] # No. of labelled samples
        u=X_u.shape[0] # No. of unlabelled samples
        
        # Variables for optimization
        h = cp.Variable()
        b = cp.Variable()
        qp = cp.Variable((l,1))
        qn = cp.Variable((l,1))
        alpha = cp.Variable((l+u,1))
        
        eta = self.opt['eta']
        eps = self.opt['eps']

        # Objective for optimization
        obj = cp.Minimize(h+(1/l)*(sum(qp)+sum(qn))+(self.opt['gamma_I']/(l+u))*cp.norm(delta@K@alpha,1))
        constraints = [(K@alpha)[0:l,:] + b*np.ones((l,1))+eta*(Y+eps)<=h, 
                       (K@alpha)[0:l,:] + b*np.ones((l,1))+eta*(Y+eps)+qp>=1,
                       -1*((K@alpha)[0:l,:] + b*np.ones((l,1))+eta*(Y-eps))<=h, 
                       -1*((K@alpha)[0:l,:] + b*np.ones((l,1))+eta*(Y-eps))+qn>=1,                       
                       qp>=0, qn>=0]


        def optimize(obj, constraints):
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.MOSEK,
                       mosek_params = {mosek.dparam.optimizer_max_time: 1000.0,
                       mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)

            # Print result.
            print("\nThe optimal value is", prob.value)
            return h.value, b.value, alpha.value
        
        print("LapMCM - Trend Filtered - Implemented in CvxPy")
        print("Current Dataset - ",self.opt['dataset'])
        tick = time.time()
        check = True
        idx = 0
        while(check and idx<2):
            try:
                h, b, beta_hat = np.array(optimize(obj, constraints), dtype=object)
                check = False
            except:
                idx = idx + 1
                print("Unexpected MOSEK error")
        tock = time.time()

        print("Time taken in optimization: ", round(tock-tick, 6), "sec")
        print("h: ",h)

        beta_hat = beta_hat.reshape(-1,)
        
        print("Num vec", np.sum(beta_hat!=0))
        
        # Computing final alpha
        self.alpha = beta_hat
        self.b = b
        
        return self.alpha, self.b, eta, self.X 

    def decision_function(self,X):
        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        eta = self.opt['eta']
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return (-1/eta)*(f+self.b)

def rbf(X1,X2,**kwargs):
    return np.exp(-cdist(X1,X2)**2*kwargs['gamma'])

# Graph difference operator
def GDO(W):
    W = W.toarray()

    n = W.shape[0]
    columns = 0
    for i in range(0, n):
        for j in range(i, n):
            if W[i, j] > 0:
                columns += 1

    delta = np.zeros((n, columns))
    m = 0
    tbool = False
    for i in range(0, n):
        for j in range(i, n):
            if W[i, j] > 0:
                if (tbool):
                    delta[i, m] = np.sqrt(W[i,j])
                    delta[j, m] = -1*np.sqrt(W[i,j])
                    tbool = False
                else:
                    delta[i, m] = -1*np.sqrt(W[i,j])
                    delta[j, m] = np.sqrt(W[i,j])  
                    tbbol = True
                m += 1
    return delta.T

def polynomial(X1,X2,**kwargs):
    return np.power((-cdist(X1,X2)**2*kwargs['gamma'] + kwargs['r']), kwargs['degree'])
