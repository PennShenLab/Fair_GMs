'''
Code for the proximal gradient method with a linear subproblem solver obtained from the following repository: https://github.com/zalgo3/zfista
'''
import time
import numpy as np
from typing import Callable, List, Optional, Tuple
from warnings import warn
import time
import random

from scipy.optimize import (
    BFGS,
    Bounds,
    LinearConstraint,
    OptimizeResult,
    minimize,
)

TERMINATION_MESSAGES = {
    0: "The maximum number of iterations is exceeded.",
    1: "Termination condition is satisfied.",
}

COLUMN_NAMES = [
    "niter",
    "nit internal",
    "max(abs(xk - yk)))",
    "subprob func",
    "learning rate",
]
COLUMN_WIDTHS = [7, 7, 13, 13, 10]
ITERATION_FORMATS = ["^7", "^7", "^+13.4e", "^+13.4e", "^10.2e"]


def _solve_subproblem(
    f: Callable,
    g: Callable,
    jac_f: Callable,
    prox_wsum_g: Callable,
    lr: float,
    lam: float,
    xk_old: np.ndarray,
    yk: np.ndarray,
    w0: Optional[np.ndarray],
    tol: float = 1e-12,
    max_iter: int = 10000,
    deprecated: bool = False,
) -> OptimizeResult:
    
    f_yk = f(yk)
    F_xk_old = f(xk_old) + g(xk_old)
    jac_f_yk = jac_f(yk)
    n_objectives = f_yk.shape[0] if isinstance(f_yk, np.ndarray) else 1

    def _dual_minimized_fun_jac(weight: np.ndarray) -> Tuple[float, np.ndarray]:
        
        wsum_jac_f_yk = np.sum(weight[:, np.newaxis, np.newaxis] * jac_f_yk, axis=0)
        yk_minus_lr_times_wsum_jac_f_yk = yk - lr * wsum_jac_f_yk
        primal_variable = prox_wsum_g(yk_minus_lr_times_wsum_jac_f_yk, lr * lam)
        primal_variable = (primal_variable + primal_variable.T)/2
        g_primal_variable = g(primal_variable)
        fun = (
            -np.inner(weight, g_primal_variable)
            - np.linalg.norm(primal_variable - yk_minus_lr_times_wsum_jac_f_yk, 'fro') ** 2
            / 2
            / lr
            + lr / 2 * np.linalg.norm(wsum_jac_f_yk, 'fro') ** 2
        )
        
        jac = -g_primal_variable - np.sum(jac_f_yk * (primal_variable - yk)[np.newaxis, :,:], axis=(1,2))
        if not deprecated:
            fun += np.inner(weight, F_xk_old - f_yk)
            jac += F_xk_old - f_yk
        return fun, jac

    res = OptimizeResult()

    res_dual = minimize(
        fun=_dual_minimized_fun_jac,
        x0=w0,
        method="trust-constr",
        jac=True,
        hess=BFGS(),
        bounds=Bounds(lb=0, ub=np.inf),
        constraints=LinearConstraint(np.ones(n_objectives), lb=1, ub=1),
        options={"gtol": tol, "xtol": tol, "barrier_tol": tol, "maxiter": max_iter},
    )
    if not res_dual.success:
        warn(res_dual.message)
    res.weight = res_dual.x
    res.x = prox_wsum_g(yk - lr * np.sum(res.weight[:, np.newaxis, np.newaxis] * jac_f_yk, axis=0), lr * lam)
    res.x = (res.x + res.x.T)/2
    res.fun = -res_dual.fun
    res.nit = res_dual.nit
    return res


def minimize_proximal_gradient(
    f: Callable,
    g: Callable,
    jac_f: Callable,
    prox_wsum_g: Callable,
    x0: np.ndarray,
    lr: float = 1e-2,
    lam: float = 1e-2,
    tol: float = 1e-5,
    tol_internal: float = 1e-12,
    max_iter: int = 10000000,
    max_iter_internal: int = 10000000,
    max_backtrack_iter: int = 100,
    warm_start: bool = False,
    decay_rate: float = 0.1,
    nesterov: bool = False,
    nesterov_ratio: Tuple[float, float] = (0, 0.25),
    return_all: bool = False,
    verbose: bool = False,
    deprecated: bool = False,
) -> OptimizeResult:
    
    if deprecated:
        warn(
            "Using the deprecated option is not mathematically proven to converge. "
            "Please consider using the recommended condition instead."
        )
    start_time = time.time()
    res = OptimizeResult(
        x0=x0,
        tol=tol,
        tol_internal=tol_internal,
        nesterov=nesterov,
        nesterov_ratio=nesterov_ratio,
    )
    if verbose:
        fmt = "|" + "|".join(["{{:^{}}}".format(x) for x in COLUMN_WIDTHS]) + "|"
        separators = ["-" * x for x in COLUMN_WIDTHS]
        print(fmt.format(*COLUMN_NAMES))
        print(fmt.format(*separators))
    xk_old = x0
    xk = x0
    yk = x0
    nit_internal = 0
    f_x0 = f(x0)
    n_objectives = f_x0.shape[0] if isinstance(f_x0, np.ndarray) else 1
    w0 = np.ones(n_objectives) / n_objectives if n_objectives > 1 else None
    if return_all:
        allvecs = [x0]
        allfuns = [f_x0 + g(x0)]
        allerrs: List[float] = []
    if nesterov:
        nesterov_tk_old = 1
    for nit in range(1, max_iter + 1):
        # print(nit)
        F_xk_old = f(xk_old) + g(xk_old)
        backtrack_iter = 0
        while True:
            try:
                subproblem_result = _solve_subproblem(
                    f,
                    g,
                    jac_f,
                    prox_wsum_g,
                    lr,
                    lam,
                    xk_old,
                    yk,
                    w0,
                    tol=tol_internal,
                    max_iter=max_iter_internal,
                    deprecated=deprecated,
                )


                xk = subproblem_result.x
                F_xk = f(xk) + g(xk)
                nit_internal += subproblem_result.nit
                if w0 is not None and warm_start:
                    w0 = subproblem_result.weight
                if verbose:
                    progress_report = [
                        nit,
                        nit_internal,
                        max(abs(xk - yk)),
                        subproblem_result.fun,
                        lr,
                    ]
                    iteration_format = ["{{:{}}}".format(x) for x in ITERATION_FORMATS]
                    fmt = "|" + "|".join(iteration_format) + "|"
                    print(fmt.format(*progress_report))
                if decay_rate == 1:
                    break
                if deprecated:
                    if np.all(f(xk) - f(yk) <= subproblem_result.fun + tol):
                        break
                elif np.all(F_xk - F_xk_old <= subproblem_result.fun + tol):
                    break
                lr *= decay_rate
                backtrack_iter += 1
                if backtrack_iter >= max_backtrack_iter:
                    raise RuntimeError(
                        "Backtracking failed to find a suitable stepsize."
                    )
            except Exception as e:
                print(f"An error occurred: {e}")
                error_res = OptimizeResult()
                error_res.success = False
                error_res.message = f"Error: {str(e)}"
                error_res.x = xk_old
                error_res.fun = F_xk_old
                error_res.nit = nit - 1
                error_res.nit_internal = nit_internal
                if return_all:
                    error_res.allvecs = allvecs
                    error_res.allfuns = allfuns
                    error_res.allerrs = allerrs
                error_res.time = time.time() - start_time
                return error_res
        error_criterion = np.max(np.abs(xk - yk))
        if return_all:
            allvecs.append(xk)
            allfuns.append(F_xk)
            allerrs.append(error_criterion)
        if error_criterion < tol:
            res.status = 1
            break
        if nesterov:
            a, b = nesterov_ratio
            nesterov_tk = np.sqrt(nesterov_tk_old**2 - a * nesterov_tk_old + b) + 0.5
            moment = (nesterov_tk_old - 1) / nesterov_tk
            yk = xk + moment * (xk - xk_old)
            nesterov_tk_old = nesterov_tk
        else:
            yk = xk
        xk_old = xk
    else:
        res.status = 0
    res.x = xk
    res.fun = F_xk
    res.success = res.status == 1
    res.message = TERMINATION_MESSAGES[res.status]
    if not res.success:
        warn(res.message)
    res.nit = nit
    # print(nit)
    res.nit_internal = nit_internal
    if return_all:
        res.allvecs = allvecs
        res.allfuns = allfuns
        res.allerrs = allerrs
    res.time = time.time() - start_time
    return res