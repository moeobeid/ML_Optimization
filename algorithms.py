# IOE 511/MATH 562, University of Michigan
# Code written by: Mouhamad Obeid

# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent
# (2) Stochastic Gradient

import numpy as np

def GDStep(w,X,y,problem,method,options):
    '''
    Function that implements gradient descent to obtain the next step
    Input: w, X, y, problem, method, options
    Output: updated value w
    '''
    g = problem.compute_g(w,X,y) #output is 2D np array dim dx1
    d = -g
    
    # Determine step size
    if method.step_type == 'Backtracking': # backtracking line search
        alpha = method.constant_step_size
        c1 = options.c1
        tau = options.tau
        while problem.compute_f(w + alpha * d,X,y) > problem.compute_f(w,X,y) + (c1 * alpha * (g.T @ d)): # performed until Armijo is not violated
            alpha = alpha * tau # update alpha until the armijo condition is satisfied
        w_new = w + alpha * d # GD formula
    
    else: # in case the user makes a typo or inserts a method.step_type not included in this file
        print('Warning: step type is not defined')

    return w_new # 1 step update value of w, dim = 2D np array dx1

def SGStep(w,X,y,k,problem,method,options):
    '''
    Function that implements gradient descent to obtain the next step
    Input: w, X, y, problem, method, options
    Output: updated value w
    '''
    # Get the batch
    batch_size = method.batch_size
    indices = np.random.choice(X.shape[0]-1, size=batch_size, replace=False)
    X_batch = X[indices, :]
    y_batch = y[indices]
    
    g = problem.compute_g(w, X_batch, y_batch)
    d = -g
    
    # Determine step size
    alpha = method.constant_step_size
    if method.step_type == 'Constant': # backtracking line search
        w_new = w + alpha * d # GD formula
    
    elif method.step_type == 'Diminishing':
        alpha = alpha / (k + 1)
        w_new = w + alpha * d # GD formula
    
    elif method.step_type == 'Backtracking': # backtracking line search
        c1 = options.c1
        tau = options.tau
        while problem.compute_f(w + alpha * d,X_batch,y_batch) > problem.compute_f(w,X_batch,y_batch) + (c1 * alpha * (g.T @ d)): # until Armijo not violated
            alpha = alpha * tau # update alpha until the armijo condition is satisfied
        w_new = w + alpha * d # GD formula
        
    else: # in case the user makes a typo or inserts a method.step_type not included in this file
        print('Warning: step type is not defined')

    return w_new, indices # 1 step update value of w, dim = 2D np array dx1