import numpy as np

from algos.GRAPH.base import base
from utils.common import soft_threshold
from utils.proximal_gradient import minimize_proximal_gradient
from utils.GRAPH.ising import objective_f_grad, loss, objectives, disparity_k_grad

class Ising(base):
    def __init__(self, T, N, lam, step_size=0.001, step_lim=0):
        super(Ising, self).__init__(T, N, lam)
        self.step_lim = step_lim
        self.step_size = step_size

 
    def multi_fair_compute(self, Ys, Y, rhom, lamm, tol):
        """
        Multi Objective Fair Ising Model
        
        Parameters:
        Ys (list): list of n_k-by-p_k group data matrix
        Y (numpy.ndarray): n-by-p data matrix
        rhom (float): trade-off hyperparameter between fairness and objective
        lamm (float): hyperparameter of l1 regularization term

        Returns:
        Theta (numpy.ndarray): Fair estimation of covariance inverse matrix
        """

        Thetas = []
        K, p = len(Ys), Y.shape[1] # group size, feature size
        Theta = self.compute(Y, history=False)

        for k in range(K):
            # local covariance inverse matrix for group k
            Thetas.append(self.compute(Ys[k], history=False))

        def multi_objective_g(X, off_diag=False): # non-differentiable part in the objective function
            if off_diag:
                res = np.ones(K+1)*np.sum(np.abs(X-np.diag(np.diag(X))))
            else:
                res = np.ones(K+1)*np.sum(np.abs(X))

            map = np.ones(len(res))*rhom
            map[0] = 1
            # return res*lamm*map
            return res*lamm
        
        def multi_objective_f(X): # differentiable part in the objective function
            res = objectives(Thetas,Ys,X,Y,False)
            map = np.ones(len(res))*rhom
            map[0] = 1

            return res*map
        
        def multi_objective_f_grad(X): # gradient of multi_objective_f(X)
            res = np.zeros((K+1, p, p))
            res[0] = objective_f_grad(X,Y)
            for k in range(K):
                res[k+1] = rhom*disparity_k_grad(Thetas,Ys,X,k,False)
            return res

        result = minimize_proximal_gradient(
            multi_objective_f,
            multi_objective_g,
            multi_objective_f_grad,
            soft_threshold,
            Theta,lam=lamm,tol=tol)
        Theta = result.x

        return Theta
    

    def compute(self, X, history=False):
        """
        Ising Model
        
        Parameters:
        X (numpy.ndarray): n-by-p binary data matrix
        history (bool): whether to record the loss history

        Returns:
        Theta (numpy.ndarray): estimation of covariance inverse matrix
        history_loss (list): history of loss
        """

        p = X.shape[1]
        
        # starting point
        Theta = np.eye(p)
        Theta_old = 2 * np.eye(p)
        
        lr = self.step_size # 0.001
        if history:
            history_loss = []
            history_loss.append(loss(Theta, X))
        for t in range(self.T):
            D = objective_f_grad(Theta, X)
            Theta_old = Theta.copy()
            
            lr = lr/np.sqrt(t+1)
            Theta = Theta - lr * D  # gradient descent
            Theta = soft_threshold(Theta, lr*self.lam) # soft threshold
            Theta = (Theta + Theta.T)/2
            if history:
                history_loss.append(loss(Theta, X))
            
            # stop criterion     
            rel_err = np.linalg.norm(Theta - Theta_old, 'fro')**2 / np.linalg.norm(Theta_old, 'fro')**2
            if rel_err < self.step_lim:
                break

        if history:
            return Theta, history_loss
        else:
            return Theta