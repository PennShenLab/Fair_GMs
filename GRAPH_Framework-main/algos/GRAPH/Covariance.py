import numpy as np
from numpy import linalg

from algos.GRAPH.base import base
from utils.common import soft_threshold
from utils.proximal_gradient import minimize_proximal_gradient
from utils.GRAPH.covariance import objective_f_grad, objective_f_cholesky, loss, objectives, disparity_k_grad

class Covariance(base):
    def __init__(self, T, N, tau, lam, ls_iter=10, step_lim=0):
        super(Covariance, self).__init__(T, N, lam)
        self.tau = tau
        self.ls_iter = ls_iter
        self.step_lim = step_lim


    def multi_fair_compute(self, Ys, Y, rhom, lamm, tol):
        """
        Multi Objective Fair Positive Definite Covariance Estimation
        
        Parameters:
        Ys (list): list of n_k-by-p_k group data matrix
        Y (numpy.ndarray): n-by-p data matrix
        rhom (float): trade-off hyperparameter between fairness and objective
        lamm (float): hyperparameter of l1 regularization term

        Returns:
        A (numpy.ndarray): Fair estimation of positive definite covariance matrix
        """

        As, Ss = [], []
        K, n, p = len(Ys), Y.shape[0], Y.shape[1] # group size, sample size, feature size
        S = Y.T @ Y / n # global covariance matrix
        A = self.compute(S)

        for k in range(K):
            n_temp = Ys[k].shape[0] # number of samples in group k
            Ss.append(Ys[k].T@Ys[k]/n_temp) # local covariance matrix of group k
            A_temp = self.compute(Ss[k]) # local estimation of positive definite covariance matrix for group k
            As.append(A_temp)

        disparity_lt = []
        for k in range(K):
            disparity_lt.append(np.abs(loss(A,Ss[k],self.tau) - loss(As[k],Ss[k],self.tau)))
        A = As[np.argmax(disparity_lt)] # starting point of fair covariance matrix

        def multi_objective_g(X, off_diag=False): # non-differentiable part in the objective function
            if off_diag:
                res = np.ones(K+1)*np.sum(np.abs(X-np.diag(np.diag(X))))
            else:
                res = np.ones(K+1)*np.sum(np.abs(X))
            map = np.ones(len(res))*rhom
            map[0] = 1
            return res*lamm
        
        def multi_objective_f(X): # differentiable part in the objective function
            res = objectives(As,Ss,X,S,self.tau)
            map = np.ones(len(res))*rhom
            map[0] = 1
            return res*map
        
        def multi_objective_f_grad(X): # gradient of multi_objective_f(X)
            res = np.zeros((K+1, p, p))
            res[0] = objective_f_grad(X,S,self.tau)
            for k in range(K):
                res[k+1] = rhom*disparity_k_grad(As,Ss,X,k,self.tau)
            return res

        result = minimize_proximal_gradient(
            multi_objective_f,
            multi_objective_g,
            multi_objective_f_grad,
            soft_threshold,A,lam=lamm,tol=tol)
        A = result.x
        return A
    

    def compute(self, S, history=False):
        """
        Positive Definite Covariance Estimation
        
        Parameters:
        S (numpy.ndarray): p-by-p covariance matrix
        history (bool): whether to record the loss history

        Returns:
        A (numpy.ndarray): estimation of precision matrix
        history_loss (list): history of loss
        """

        # starting point
        A_diag = self.lam*np.ones(self.N)
        A_diag = A_diag + np.diag(S)
        A_diag = 1.0 / A_diag
        A = np.diag(A_diag)

        history_loss = []
        if history:
            history_loss.append(loss(A, S, self.tau))

        init_step = np.float32(1.0)
        A_inv = linalg.inv(A)
        for _ in range(self.T):
            
            # linesearch for step size
            A_next, step = linesearch(A, S, self.tau, self.lam, max_iter=self.ls_iter, init_step=init_step,
                                            step_lim=self.step_lim)
            if step == 0:
                break
            else:
                # Barzilai-Borwein method for a new initial step size
                A_next_inv = linalg.inv(A_next)
                A_next_A = A_next - A

                # stop criterion
                if np.sum(np.abs(A_next_A)) < 1e-10:
                    break

                init_step = np.sum(np.square(A_next_A))
                div_init_step = np.trace((A_next_A) @ (A_inv - A_next_inv))
                A_next_A = None
                if div_init_step != 0:
                    init_step /= div_init_step
                else:
                    init_step = 0
                A = A_next
                A_next = None
                A_inv = A_next_inv
                A_next_inv = None

            if history:
                history_loss.append(loss(A, S, self.tau))

        if history:
            return A, history_loss
        else:
            return A


def objective_Q(objective_f_value, A, D, A_next, step):
    A_next_A = A_next - A
    return objective_f_value+np.trace(A_next_A@D)+(0.5/step)*(np.sum(np.square(A_next_A)))


def linesearch(A, S, tau, lam, max_iter, init_step, step_lim):
    if init_step == 0:
        return A, 0.0
    step = init_step
    D = objective_f_grad(A,S,tau)
    L = linalg.cholesky(A)
    init_F_value = objective_f_cholesky(A,S,L,tau)
    L = None
    for _ in range(max_iter):
        if step < step_lim: break
        try:
            A_next = soft_threshold(A - step * D, step * lam)
            A_next = A_next + np.transpose(A_next)
            A_next *= 0.5
            L_next = linalg.cholesky(A_next)
            if objective_f_cholesky(A_next, S, L_next, tau) <= objective_Q(init_F_value, A, D, A_next, step):
                return A_next, step
        except linalg.LinAlgError:
            pass
        step *= 0.5
    step = linalg.eigvalsh(A)[0] ** 2
    A_next = soft_threshold(A - step * D, step * lam)
    A_next = A_next + np.transpose(A_next)
    A_next *= 0.5
    try:
        L_next = linalg.cholesky(A_next)
    except linalg.LinAlgError:
        step = 0.0
        A_next = A
    return A_next, step