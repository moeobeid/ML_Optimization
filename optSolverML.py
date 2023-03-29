# IOE 511/MATH 562, University of Michigan
# Code written by: Mouhamad Obeid

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (w) and final function value (f)

import numpy as np
from scipy.io import loadmat

import functions
import algorithms

def optSolverML_Obeid_Mouhamad(problem,method,options):
    
    dataset_name = problem.dataset_name
    name = problem.name
    
    def loadData(dataset_name):
        data = loadmat(f'./data/{dataset_name}.mat')
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        f_star = loadmat(f'./data/f_star_{name}_{dataset_name}.mat')['f_star']
        return X_train, y_train, X_test, y_test, f_star
    
    X_train, y_train, X_test, y_test, f_star = loadData(dataset_name)
    w = np.zeros((X_train.shape[1],1)) #get w0
   
    # Method name to implement the desired algorithm
    if method.name == 'GradientDescent':
        n = 1
        grad_eval = 20 * n
        y_pred_train = np.zeros((y_train.shape[0], grad_eval))
        y_pred_test = np.zeros((y_test.shape[0], grad_eval))
        acc_tr = np.zeros(grad_eval)
        acc_te = np.zeros(grad_eval)
        f_tr = np.zeros(grad_eval)
        f_te = np.zeros(grad_eval)
        
        for i in range(20 * n):
            w_new = algorithms.GDStep(w,X_train,y_train,problem,method,options) 
            w = w_new
            
            f_tr[i] = problem.compute_f(w,X_train,y_train)
            f_te[i] = problem.compute_f(w,X_test,y_test)
            
            y_pred_train_i = X_train @ w
            y_pred_train[:, i] = np.where(y_pred_train_i.ravel() >= 0, 1, -1)
            y_pred_test_i = X_test @ w
            y_pred_test[:, i] = np.where(y_pred_test_i.ravel() >= 0, 1, -1)
            
            acc_tr[i] = np.sum(y_pred_train[:, i] == y_train.ravel()) #/ y_train.shape[0]
            acc_te[i] = np.sum(y_pred_test[:, i]  == y_test.ravel()) #/ y_test.shape[0]
    
    elif method.name == 'StochasticGradient':
        n = X_train.shape[0] + X_test.shape[0]
        batch_size = method.batch_size
        grad_eval = (20 * n) // batch_size
        
        y_pred_train = np.zeros((y_train.shape[0], grad_eval))
        y_pred_test = np.zeros((y_test.shape[0], grad_eval))
        acc_tr = np.zeros(grad_eval)
        acc_te = np.zeros(grad_eval)
        f_tr = np.zeros(grad_eval)
        f_te = np.zeros(grad_eval)
        
        for k in range(grad_eval):
            w_new, indices = algorithms.SGStep(w,X_train,y_train,k,problem,method,options) 
            w = w_new
            
            f_tr[k] = problem.compute_f(w,X_train,y_train)
            f_te[k] = problem.compute_f(w,X_test,y_test)

            y_pred_train_k = X_train @ w
            y_pred_train[:, k] = np.where(y_pred_train_k.ravel() >= 0, 1, -1)
            y_pred_test_k = X_test @ w
            y_pred_test[:, k] = np.where(y_pred_test_k.ravel() >= 0, 1, -1)

            acc_tr[k] = np.sum(y_pred_train[:, k] == y_train.ravel()) #/ y_train[indices].shape[0]
            acc_te[k] = np.sum(y_pred_test[:, k]  == y_test.ravel()) #/ y_test.shape[0]
    else:
        print('Warning: method is not implemented yet')
    
    f_tr = f_tr - f_star[0]
    return w[-1],f_tr[-1],acc_tr[-1],f_te[-1],acc_te[-1]