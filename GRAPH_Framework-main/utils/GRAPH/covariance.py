import numpy as np

def objective_f_grad(A,S,tau):
    A_inv = np.linalg.inv(A)
    return A-S-tau*A_inv

def loss(A,S,tau):
    t1 = np.linalg.norm(A-S,'fro')**2/2
    t2 = -tau*np.log(np.linalg.det(A))
    return t1 + t2

def loss_l1(A,S,tau,lam,off_diag=False):
    if off_diag:
        return loss(A,S,tau)+lam*np.sum(np.abs(A-np.diag(np.diag(A))))
    else:
        return loss(A,S,tau)+lam*np.sum(np.abs(A))

def disparity(As,A,Xs,tau,lam,withlam=False):
    res = 0
    n = len(As)
    N = np.array([Xs[i].shape[0] for i in range(n)])
    Ss = np.array([Xs[i].T@Xs[i]/N[i] for i in range(n)])
    for k in range(n):
        if withlam:
            t1 = loss_l1(A,Ss[k],tau,lam) - loss_l1(As[k],Ss[k],tau,lam)
        else:
            t1 = loss(A,Ss[k],tau) - loss(As[k],Ss[k],tau)
        for s in range(n):
            if s == k:
                continue
            if withlam:
                t2 = loss_l1(A,Ss[s],tau,lam) - loss_l1(As[s],Ss[s],tau,lam)
            else:
                t2 = loss(A,Ss[s],tau) - loss(As[s],Ss[s],tau)
            res += 1/2*(t1 - t2)**2
    return res

def objectives(As,Ss,A,S,tau,disparity=False):
    n = len(As)
    res = np.zeros(n+1)
    res[0] = loss(A,S,tau)
    for k in range(n):
        t1 = loss(A,Ss[k],tau) - loss(As[k],Ss[k],tau)
        if disparity:
            res[k+1] = 1/2*t1**2
        else:
            for s in range(n):
                if s == k:
                    continue
                t2 = loss(A,Ss[s],tau) - loss(As[s],Ss[s],tau)
                res[k+1] += 1/2*(t1 - t2)**2
    return res

def disparity_k_grad(As,Ss,A,k,tau,disparity=False):
    n = len(As)
    res = 0
    t1 = objective_f_grad(A,Ss[k],tau)
    t2 = loss(A,Ss[k],tau) - loss(As[k],Ss[k],tau)
    if disparity:
        return t2*t1
    else:
        for s in range(n):
            t3 = objective_f_grad(A,Ss[s],tau)
            t4 = loss(A,Ss[s],tau) - loss(As[s],Ss[s],tau)
            res += (t2 - t4)*(t1 - t3)
        return res

def objective_f_cholesky(A,S,L,tau):
    t1 = -2*tau*np.sum(np.log(np.diagonal(L),dtype='float32'))
    t2 = np.linalg.norm(A-S,'fro')**2/2
    return t1 + t2

def objective_F_cholesky(A,S,lam,L,tau):
    return objective_f_cholesky(A,S,L,tau)+lam*np.sum(np.abs(A))