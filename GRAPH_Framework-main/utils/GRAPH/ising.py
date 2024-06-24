import numpy as np

def objective_f_grad(Theta, X):
    n, p = X.shape
    grad = np.zeros((p, p))
    temp = np.zeros((p, n))

    Xhat = X.T@X

    temp = np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T) / \
           (1 + np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T))

    for k in range(p):
        grad[k, :] = temp[k, :] @ X
        grad[k, k] = np.sum(temp[k, :])
        
    return (grad - Xhat) / n

def loss(Theta,X):
    n, p = X.shape
    Xhat = X.T@X
    loss = -np.sum(Xhat*Theta)
    loss += np.sum(np.log(1 + np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T)))
    return loss / n

# def objective_f_grad(Theta, X):
#     n, p = X.shape
#     grad = np.zeros((p, p))
#     temp = np.zeros((p, n))

#     Xhat = X.T@X

#     temp = np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T) / \
#            (1 + np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T))

#     for k in range(p):
#         grad[k, :] = -Xhat[k, :] - Xhat[:, k] + \
#                      np.sum(np.tile(np.exp(Theta[k, k] + Theta[k, :].reshape(1, -1) @ X.T -
#                                            Theta[k, k] * X[:, k].reshape(1, -1)) /
#                                     (1 + np.exp(Theta[k, k] + Theta[k, :].reshape(1, -1) @ X.T -
#                                                 Theta[k, k] * X[:, k].reshape(1, -1))), (p, 1)) * X.T, axis=1) + \
#                      np.sum(temp * np.tile(X[:, k], (p, 1)), axis=1)

#         grad[k, k] = -Xhat[k, k] + np.sum(np.exp(Theta[k, k] + Theta[k, :].reshape(1, -1) @ X.T -
#                                                  Theta[k, k] * X[:, k].reshape(1, -1)) /
#                                           (1 + np.exp(Theta[k, k] + Theta[k, :].reshape(1, -1) @ X.T -
#                                                       Theta[k, k] * X[:, k].reshape(1, -1))))
        
#     return grad / n

# def loss(Theta,X):
#     n, p = X.shape
#     Xhat = X.T@X
#     loss = -np.sum(Xhat*Theta)
#     loss += np.sum(np.log(1 + np.exp(np.diag(Theta).reshape(p, 1) @ np.ones((1,n)) + Theta @ X.T - np.diag(np.diag(Theta)) @ X.T)))
#     return loss / n

def loss_l1(Theta,X,lam,off_diag=False):
    if off_diag:
        return loss(Theta,X)+lam*np.sum(np.abs(Theta-np.diag(np.diag(Theta))))
    else:
        return loss(Theta,X)+lam*np.sum(np.abs(Theta))

def disparity(Thetas,Theta,Xs,lam,withlam=False):
    res = 0
    n = len(Thetas)
    for k in range(n):
        if withlam:
            t1 = loss_l1(Theta,Xs[k],lam) - loss_l1(Thetas[k],Xs[k],lam)
        else:
            t1 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
        for s in range(n):
            if s == k:
                    continue
            if withlam:
                t2 = loss_l1(Theta,Xs[s],lam) - loss_l1(Thetas[s],Xs[s],lam)
            else:
                t2 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
            res += 1/2*(t1 - t2)**2
    return res

def objectives(Thetas,Xs,Theta,X,disparity=False):
    n = len(Thetas)
    res = np.zeros(n+1)
    res[0] = loss(Theta,X)
    for k in range(n):
        t1 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
        if disparity:
            res[k+1] = 1/2*t1**2
        else:
            for s in range(n):
                if s == k:
                    continue
                t2 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
                res[k+1] += 1/2*(t1 - t2)**2
    return res

def disparity_k_grad(Thetas,Xs,Theta,k,disparity=False):
    n = len(Thetas)
    res = 0
    t1 = objective_f_grad(Theta,Xs[k])
    t2 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
    if disparity:
        return t2*t1
    else:
        for s in range(n):
            t3 = objective_f_grad(Theta,Xs[s])
            t4 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
            res += (t2 - t4)*(t1 - t3)    
        return res
    
# def disparity(Thetas,Theta,Xs,lam,withlam=False):
#     res = 0
#     n = len(Thetas)
#     for k in range(n):
#         if withlam:
#             t1 = loss_l1(Theta,Xs[k],lam) - loss_l1(Thetas[k],Xs[k],lam)
#         else:
#             t1 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
#         for s in range(n):
#             if s == k:
#                     continue
#             if withlam:
#                 t2 = loss_l1(Theta,Xs[s],lam) - loss_l1(Thetas[s],Xs[s],lam)
#             else:
#                 t2 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
#             res += np.exp(t1 - t2)
#     return res

# def objectives(Thetas,Xs,Theta,X,disparity=False):
#     n = len(Thetas)
#     res = np.zeros(n+1)
#     res[0] = loss(Theta,X)
#     for k in range(n):
#         t1 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
#         if disparity:
#             res[k+1] = np.exp(t1)
#         else:
#             for s in range(n):
#                 if s == k:
#                     continue
#                 t2 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
#                 res[k+1] += np.exp(t1 - t2)
#     return res

# def disparity_k_grad(Thetas,Xs,Theta,k,disparity=False):
#     n = len(Thetas)
#     res = 0
#     t1 = objective_f_grad(Theta,Xs[k])
#     t2 = loss(Theta,Xs[k]) - loss(Thetas[k],Xs[k])
#     if disparity:
#         return np.exp(t2)*t1
#     else:
#         for s in range(n):
#             t3 = objective_f_grad(Theta,Xs[s])
#             t4 = loss(Theta,Xs[s]) - loss(Thetas[s],Xs[s])
#             res += np.exp(t2 - t4)*(t1 - t3)    
#         return res