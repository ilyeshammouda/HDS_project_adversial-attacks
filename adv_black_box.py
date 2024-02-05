"""

This code was inspired from the article: https://arxiv.org/abs/2003.13001
HanQin Cai, Daniel Mckenzie, Wotao Yin, and Zhenliang Zhang. Zeroth-Order Regularized Optimization (ZORO):
Approximately Sparse Gradients and Adaptive Sampling. arXiv preprint arXiv: 2003.13001.
As well as their git repo: https://github.com/caesarcai/ZORO
Some changes have been made to test new algorithms that can be more adaptive than the ones presented in the article.
"""
import numpy as np
import numpy.linalg as la
from interface import BaseOptimizer
from Cosamp import cosamp
from help_function import ISTA_ad,IHT_ad,IHT_classique,debiased_Lasso,Lasso_reg,True_grad_SparseQuadric,True_grad_square_of_the_difference_support_S,True_grad_norm_with_a_Gaussian_matrix
import projection as proj
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision






class ZORO_nest(BaseOptimizer):
    '''
    ZORO for black box optimization.
    '''
    def __init__(self, img,f, params,device ,threshold_IHT=2,function_budget=10000,
                 function_target=None,s=20,step_IHT=0.0000001,itt_IHT=30,C_IHT=0.9,lamda_IHT=0.1,epsilon=0,lmax=20,r=3,x_star=0):

        super().__init__()
        if r < 3:
            # According to the article r should be greater or equal to 3
            warnings.warn("The value of 'r' should be greater than or equal to 3. Please set 'r' to a value greater than 3.", UserWarning)


        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.img = img
        self.img_numpy=img.numpy()
        self.d=self.img_numpy.shape
        self.img_vect=self.img_numpy.flatten()
        self.n = len(self.img_vect)
        self.noise=np.zeros(self.n)
        self.t = 0
        self.delta = params["delta"]
        self.step_size = params["step_size"]
        self.num_samples = params["num_samples"]
        self.s=s
        self.step_IHT=step_IHT
        self.itt_IHT=itt_IHT
        self.threshold_IHT=threshold_IHT
        self.C_IHT=C_IHT
        self.lamda_IHT=lamda_IHT
        self.epsilon=epsilon
        self.r=r
        self.lmax=lmax
        self.device=device

        #self.M=params["M"]


        # Define sampling matrix
        self.Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        #cosamp_params = { "Z": Z,"delta": self.delta, "maxiterations": 10,
                        # "tol": 0.5, "sparsity": self.sparsity}
        #self.cosamp_params = cosamp_params

    def nastrov_step(gradient_t,x_t,y_t_1,lamda_t,gamma_t,alpha=0.05):
        '''
        Nastrov step
        '''
        y_t=y_t_1-alpha*gradient_t
        x_t_plus_1=(1-gamma_t)*x_t+gamma_t*y_t
        lamda_t_plus_1=(1 + math.sqrt(1 + 4 * (lamda_t**2))) / 2
        gamma_t_plus_1=(1-lamda_t)/lamda_t_plus_1
        return x_t_plus_1,y_t,lamda_t_plus_1,gamma_t_plus_1



    """
    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
    """



    def GradEstimate(self):
        '''
        Gradient estimation sub-routine.
        '''
        Z = self.Z
        delta = self.delta
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1
        num_samples = np.size(Z, 0)
        f = self.f
        f.eval()
        y = np.zeros(num_samples)
        function_estimate = 0
        target=self.noise+self.img_vect
        #ad_attack=self.img_vect+self.noise
        for i in range(num_samples):
            y_1= target + delta*np.transpose(Z[i,:])
            y_2=target - delta*np.transpose(Z[i,:])
            y_1=y_1.reshape(self.d)
            y_2=y_2.reshape(self.d)
            y_1=(torch.from_numpy(y_1).float()).to(self.device)
            y_2=(torch.from_numpy(y_2).float()).to(self.device)
            y_temp = f(y_1)
            y_temp3=f(y_2)

            y_temp=torch.softmax(y_temp, dim=-1)
            y_temp3=torch.softmax(y_temp3, dim=-1)
            topk_vals, topk_idx = y_temp.topk(1, dim=-1)
            topk_vals_3, topk_idx_3 = y_temp3.topk(1, dim=-1)
            y_temp = topk_vals.cpu().numpy()
            y_temp3 = topk_vals_3.cpu().numpy()

            #y_temp2 = f(x)
            #function_estimate += y_temp2
            y[i] = (y_temp - y_temp3)/(2*np.sqrt(num_samples)*delta)

            self.function_evals += 2
        function_estimate= f((torch.from_numpy(self.noise.reshape(self.d)).float()).to(self.device))
        #function_estimate = function_estimate/num_samples
        #Z = Z/np.sqrt(num_samples)
        #y=(y.numpy()).flatten()

        grad_estimate=IHT_ad(X=Z,Y=y,threshold=self.threshold_IHT,C=self.C_IHT,step=self.step_IHT,max_iterations=self.itt_IHT,lamda=self.lamda_IHT)
        print('grad_estimated')
        print(grad_estimate)
        return grad_estimate,function_estimate



    def step(self):
        '''
        Take step of optimizer
        '''

        grad_est,f_est = self.GradEstimate()
        self.fd = f_est
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        #noise_plus_one,y,lamda,gamma=ZORO_nest.nastrov_step(gradient_t=grad_est,x_t=self.noise,y_t_1=y,lamda_t=lamda,gamma_t=gamma)
        self.noise= ( self.noise-self.step_size*grad_est) # gradient descent

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals,'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, 'T'

        self.t += 1
        return self.function_evals, False



    def Zoro(self):
        #performance_log_ZORO = [[0, self.f(self.x)]]
        #cost_x=[[0,np.linalg.norm(self.x-self.x_star)]]
        termination = False
        #lamda_prev=0
        #y_n_prev=self.noise
        i=0
        while termination is False:
            #evals_ZORO, solution_ZORO, termination,y_cur,lamda_cur,gamma_cur,noise_plus_one = self.step(y=y_n_prev,lamda=lamda_prev,gamma=gamma_prev)
            #y_n_prev=y_cur
            #lamda_prev=lamda_cur
            #gamma_prev=gamma_cur
            i=0+1
            _,termination=self.step()
            print(f'noise{self.noise}')
            print(f'itt{i}')
            #cost=np.linalg.norm(self.x-self.x_star)

            # save some useful values
            #performance_log_ZORO.append( [evals_ZORO,np.mean(self.fd)] )
            #cost_x.append([evals_ZORO,cost])

            # print some useful values
            #performance_log_ZORO.append( [evals_ZORO,self.f(solution_ZORO)] )
            #self.report( 'Estimated f(x_k): %f norm of the estimated gradient: %f  function evals: %d Norm True grad: %f \n' %
        return(self.noise)



