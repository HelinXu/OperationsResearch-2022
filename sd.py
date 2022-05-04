# steepest descent method

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt


def f(x):
    '''
    object function
    '''
    return (1-x[0])**2+2*(x[0]**2-x[1])**2

def df(x, ord=2):
    '''
    return the normalized gradient direction of f at x
    '''
    df0 = -2 + 2*x[0] + 8*x[0]**3 - 8*x[0]*x[1]
    df1 = -4*x[0]**2 + 4*x[1]
    if ord == 1 or ord == '1':
        # 最速下降方向，1范数
        if abs(df0) > abs(df1):
            return np.array([np.sign(-df0), 0])
        else:
            return np.array([0, np.sign(-df1)])
    elif ord == 2 or ord == '2':
        # 最速下降方向，2范数
        d = np.array([-df0, -df1])
        return d / np.linalg.norm(d, ord=2)
    else:
        # 最速下降方向，inf范数
        return np.array([np.sign(-df0), np.sign(-df1)])


def sd(f, df, x0, eps=1e-4, max_iter=100, ord=2):
    '''
    steepest descent method
    TODO: eps
    '''
    guesses = [x0]
    while len(guesses) < max_iter:
        x = guesses[-1]
        D = df(x, ord=ord)

        def f1d(alpha):
            return f(x + alpha * D)

        alpha_opt = sopt.golden(f1d, tol=eps) # 精确直线搜索，终止条件为tol

        next_guess = x + alpha_opt * D
        guesses.append(next_guess)
        print(next_guess)
    return guesses