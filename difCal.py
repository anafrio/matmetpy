#Classes and functions concerning non linear systems of equations

import scipy

def FunDerivative(fun, h=0.1e-5):
    '''
    Returns function derivative for fun
    '''
    def df(x):
        return ( fun(x+h/2) - fun(x-h/2) )/h
    return df

def NumDerivative(fun, x, h=0.1e-5):
    '''
    Computes numerical approximation of a functions derivative at point x
    '''
    df = FunDerivative(fun,h)
    return df(x)


def NumJacobian(vecFun, X, h=0.1e-5):
    '''
    Computes numerical approximation of a vector function jacobian at point X. The code object vecFun must recieve an array of length len(X) as (the only) argument.
    '''
    n = len(X)
    m = len(vecFun(X))
    jac = scipy.zeros((n,m), float)
    hv = scipy.zeros(n, float)
    for i in range(n):
       hv[i] = h
       jac[i] = (scipy.array(vecFun(X+hv/2)) - scipy.array(vecFun(X-hv/2)))/h
       hv[i] = 0.0
    return jac.transpose()

   
