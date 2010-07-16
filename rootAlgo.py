# Functions implementing root finding algorithms

import scipy
import scipy.linalg
import difCal

def ScaNewRapSolve(fun, x0, funD = None, tol=0.5e-5, miniter = 1, maxiter=100, h =0.1e-5):
    '''
    Uses Newton Raphson algorithm to calculate a root x of the given scalar 
    function: fun(x) = 0
    '''
  
    #Initializations
    i = 0
    newx = x0
    funEva = fun(newx)
    if funD == None:
        derEva = difCal.NumDerivative(fun,newx,h)
    else:
        derEva = funD(newx)
             
    # Iterate for i less than maxiter and fun(x)>tol
    while(i<miniter or (i<maxiter and abs(funEva)>tol)):
        
        newx = newx - funEva/derEva
        funEva = fun(newx)
        if funD == None:
            derEva = difCal.NumDerivative(fun,newx,h)
        else:
            derEva = funD(newx)
        i = i+1
        #print '[ScaNewRapSolve] Iteration ', str(i),', newx = ', str(newx),', funEva = ', str(funEva)
    if i == maxiter:
        raise ValueError, '[ScaNewRapSolve] Convergence not attained for the initial value, tolerance and maxiter given'
    else:
        return newx
    
def VecNewRapSolve(funVec, X0, funjac = None, tol=0.1e-5, miniter = 1, maxiter=100, h =0.1e-5):
    '''
    Uses Newton Raphson algorithm to calculate a vector root X of the given vector 
    function: funVec(X) = (0,...,0)
    '''
    #Initializations
    i=0
    newX = scipy.array(X0)
    funVecEva = funVec(newX)
    if funjac == None:
        jacEva = difCal.NumJacobian(funVec,newX,h)
    else:
        jacEva = funjac(newX)
        #Check for correct size relations for X0 and jac if given
        if (len(X0),len(X0)) != jacEva.shape:
            raise ValueError, '[VecNewRapSolve] Sizes of the jacobian given are not correct'

    tolerr = False
    for i in range(len(funVecEva)):
        if abs(funVecEva[i]) > tol:
            tolerr = True
        
    i=0
    # Iterate for i less than maxiter and funVec(X)>tol
    while(i<miniter or (i<maxiter and tolerr)):
        if scipy.linalg.det(jacEva) == 0.:
            raise ValueError, '[VecNewRapSolve] Non invertible jacobian encountered. Try a different initial value'
        else:
            invJac = scipy.linalg.inv(jacEva)
            newXlast = newX
            newX = newXlast - scipy.dot(invJac,funVecEva)
            funVecEva = scipy.array(funVec(newX))
            if funjac == None:
                jacEva = difCal.NumJacobian(funVec,newX,h)
            else:
                jacEva = funjac(newX)
            i = i+1
            #print '[VecNewRapSolve] Iteration ', str(i),', newX = ', str(newX),', funEva = ', funVecEva,'jacEva=',jacEva
            tolerr = False
            for j in range(len(funVecEva)):
                if abs(funVecEva[j]) > tol:
                    tolerr = True
    if i == maxiter:
        raise ValueError, '[VecNewRapSolve] Convergence not attained for the initial value, tolerance and maxiter given'
    else:
        return newX