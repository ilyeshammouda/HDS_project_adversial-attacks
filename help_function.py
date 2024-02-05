""" 
Created on Mon Jul 11:47:35 2023
@ author: ilyeshammouda
This code defines some useful functions that will be used 
"""

import numpy as np
from sklearn import linear_model
import random
from math import log,sqrt
from scipy.optimize import minimize
import numpy as np
from scipy.sparse import csr_matrix

def Lasso_reg(Y,Z,debiased='False',cv=5,it=100,tol=0.0001):
        """
        This function gives the solution to the Lasso regression in a multivariate model
        """
        lasso = linear_model.LassoCV(cv=cv,max_iter=it,tol=tol)
        lasso.fit(Z,Y)
        g = lasso.coef_
        u=lasso.intercept_
        if debiased=='False':
                return(g)
        else:
                return(g,u)


def debiased_Lasso(Y,Z,delta,cv=5,it=100,tol=0.0001):
        n=Z.shape[0]
        ones=np.ones(n)
        g_t,u=Lasso_reg(Y,Z,debiased='True',cv=cv,it=it,tol=tol)
        y_tilde=Y
        g_tilde=g_t+((1/n)*Z.T@(y_tilde-u*ones))
        return(g_tilde)



def s_grands_comp(vecteur, s):
    vecteur_np = np.array(vecteur)
    s_largest_indices = np.abs(vecteur_np).argsort()[-s:][::-1]
    s_plus_larges = np.zeros_like(vecteur_np)
    s_plus_larges[s_largest_indices] = vecteur_np[s_largest_indices]
    
    return s_plus_larges

def IHT_classique(X, Y,s,step=0.00000001,max_iterations=30):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)     
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=s_grands_comp(Z, s)
        return beta_hat


def HardThreshold(x,lamda):
        return x*(np.abs(x)>=lamda)


def SoftThreshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
def IHT_ad(X, Y,threshold ,C=0.9,step=0.0000001,max_iterations=30,lamda=0.1):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)  
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=HardThreshold(Z, lamda)      
            if lamda > threshold:
                lamda*=C
        return beta_hat


def ISTA_ad(X, Y,threshold ,C=0.9,step=0.0000001,max_iterations=30,lamda=0.1):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)  
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=SoftThreshold(Z, lamda)
            while lamda > threshold:
                lamda*=C
        return beta_hat




class tools:
    def Rademacher_matrix(d,n):
        """
        This fucntion generates a Rademacher matrix
        """
        return np.random.choice([-1, 1], size=(d,n))



    def Rademacher_matrix_concatenated(d,n):
        """
        This function generates a Rademacher matrix and add a line of ones 
        """
        Z=tools.Rademacher_matrix(d,n)
        Last_line_of_ones = np.ones((1, Z.shape[1]))
        return (Z,np.concatenate((Z, Last_line_of_ones), axis=0))
    

    def sparse_vector(n, s):
        non_zero_indices = np.random.choice(range(n), size=s, replace=False)
        non_zero_values = np.random.rand(s)
        # Create the sparse vector using the CSR (Compressed Sparse Row) format
        sparse_vector = np.zeros(n)
        sparse_vector[non_zero_indices] = non_zero_values
        return sparse_vector
    

    def vecteur_sparse_support_S(n,s):
        if s <= 0 or s > n:
            raise ValueError("Le nombre de composantes non nulles doit être compris entre 1 et la taille du vecteur.")
        zeros=np.zeros(n)
        for i in range(s):
            zeros[i]=np.random.normal(0,1)
        return(zeros)
    



class algos:
    def Lasso_reg(Y_tilde,Z,lamda=0.1):
        """
        This function gives the solution to the Lasso regression in a multivariate model
        """
        lasso = linear_model.Lasso(lamda)
        lasso.fit(Z,Y_tilde )
        g = lasso.coef_
        u=lasso.intercept_
        return(g,u)


    def GradiantEstimate(y_t_vect,d,n,delta,lamda=0.1):
        """ 
        This function corresponds to the pseudo algorithme 1 defined in the paper
        """
        Z=tools.Rademacher_matrix(n,d)
        y_tilde=y_t_vect/delta
        (g,u)=algos.Lasso_reg(Y_tilde=y_tilde,Z=Z,lamda=lamda)
        return(g,u)
    

        
class Tests:
    # Functions that will be used for testing 
    def f_test(x_t,delta,d,noise=1):
        Z=tools.Rademacher_matrix(d,1)
        return(np.linalg.norm(x_t+delta*Z)+noise*np.random.normal(0,1,1))
    def vect_f_test(x_t, delta, d, n):
        y_t_vecteur = np.zeros(n)
        for i in range(n):
            y_t_vecteur[i] = Tests.f_test(x_t, delta, d)
        y_t = Tests.f_test(x_t=x_t, delta=0, d=d, noise=0)
        return y_t_vecteur, y_t



def True_grad_SparseQuadric(x,s):
     
     grad=np.zeros(len(x))
     grad[0:s]=x[0:s]
     return(2*grad)


def True_grad_square_of_the_difference_support_S(x,x_star,s):
     grad=np.zeros(len(x))
     grad[0:s]=(x-x_star)[0:s]
     return(2*grad)

def True_grad_norm_with_a_Gaussian_matrix(x,x_star,A):
    diag_A = np.diagonal(A)
    matrix_diag_A = np.diag(diag_A)
    part1=2*(matrix_diag_A@(x-x_star))       
    part2=A@(x-x_star)
    return(part1+part2)



     
