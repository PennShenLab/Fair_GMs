import numpy as np

def objective_f_grad(A_inv,S):
    return S-A_inv
    
def loss(A,S):
    t1 = -np.log(np.linalg.det(A))
    t2 = np.trace(S@A)
    return t1+t2

def loss_l1(A,S,lam,off_diag=False):
    if off_diag:
        return loss(A,S)+lam*np.sum(np.abs(A-np.diag(np.diag(A))))
    else:
        return loss(A,S)+lam*np.sum(np.abs(A))

def disparity(As,A,Xs,lam,withlam=False):
    res = 0
    n = len(As)
    N = np.array([Xs[i].shape[0] for i in range(n)])
    Ss = np.array([Xs[i].T@Xs[i]/N[i] for i in range(n)])
    for k in range(n):
        if withlam:
            t1 = loss_l1(A,Ss[k],lam) - loss_l1(As[k],Ss[k],lam)
        else:
            t1 = loss(A,Ss[k]) - loss(As[k],Ss[k])
        for s in range(n):
            if s == k:
                    continue
            if withlam:
                t2 = loss_l1(A,Ss[s],lam) - loss_l1(As[s],Ss[s],lam)
            else:
                t2 = loss(A,Ss[s]) - loss(As[s],Ss[s])
            res += 1/2*(t1 - t2)**2
    return res

def objectives(As,Ss,A,S,disparity=False):
    n = len(As)
    res = np.zeros(n+1)
    res[0] = loss(A,S)
    for k in range(n):
        t1 = loss(A,Ss[k]) - loss(As[k],Ss[k])
        if disparity:
            res[k+1] = 1/2*t1**2
        else:
            for s in range(n):
                if s == k:
                    continue
                t2 = loss(A,Ss[s]) - loss(As[s],Ss[s])
                res[k+1] += 1/2*(t1 - t2)**2
    return res

def disparity_k_grad(As,Ss,A,k,disparity=False):
    n = len(As)
    res = 0
    t1 = objective_f_grad(np.linalg.inv(A),Ss[k])
    t2 = loss(A,Ss[k]) - loss(As[k],Ss[k])
    if disparity:
        return t2*t1
    else:
        for s in range(n):
            t3 = objective_f_grad(np.linalg.inv(A),Ss[s])
            t4 = loss(A,Ss[s]) - loss(As[s],Ss[s])
            res += (t2 - t4)*(t1 - t3)    
        return res

def objective_f_cholesky(A,S,L):
    t1 = -2*np.sum(np.log(np.diagonal(L),dtype='float32'))
    t2 = np.trace(S@A,dtype='float32')
    return t1 + t2

def objective_F_cholesky(A,S,lam,L):
    return objective_f_cholesky(A,S,L)+lam*np.sum(np.abs(A))