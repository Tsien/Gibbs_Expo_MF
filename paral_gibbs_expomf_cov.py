"""

Exposure Matrix Factorization with exposure covariates (e.g. topics, or
locations) for collaborative filtering

This file is largely adapted from expomf.py

"""

import pdb
import scipy.stats
import os
import glob
import sys
import time
import numpy as np
from numpy import linalg as LA
from numpy import ones
from scipy import sparse
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval

floatX = np.float32
EPS = 1e-8


class ExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, min_chunk_count = 20, n_components=100, max_iter=10, batch_size=1000,
                 batch_sgd=10, max_epoch=10, init_std=0.01, n_jobs=8,
                 random_state=None, save_params=False, save_dir='.',
                 early_stopping=False, verbose=False, **kwargs):
        '''
        Exposure matrix factorization

        Parameters
        ---------
        min_chunk_count : int
            Minimum number of parallel chunks in Gibbs Sampling
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        batch_sgd: int
            Batch size for SGD when updating exposure factors
        max_epoch: int
            Number of epochs for SGD
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.min_chunk_count = min_chunk_count
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.batch_sgd = batch_sgd
        self.max_epoch = max_epoch
        self.init_std = init_std
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.mse = ones((max_epoch, 2))

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters

        Parameters
        ---------
        lambda_theta, lambda_beta, lambda_nu: float
            Regularization parameter for user (lambda_theta), item CF factors (
            lambda_beta) and user exposure factors (lambda_nu). Default value
            1e-5. Since in implicit feedback all the n_users-by-n_items data
            points are used for training, overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        learning_rate: float
            Learning rate for SGD. Default value 0.1. Since for each user we
            are effectively doing logistic regression, constant learning rate
            should suffice
        init_mu: float
            init_mu is used to initalize the user exposure bias (alpha) such
            that all the \mu_{ui} = inv_logit(nu_u * pi_i + alpha_u) is roughly
            init_mu. Default value is 0.01. This number should change according
            to the sparsity of the data (sparser data with smaller init_mu).
            In the experiment, we select the value from validation set
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_nu = float(kwargs.get('lambda_nu', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.learning_rate = float(kwargs.get('learning_rate', 0.1))
        self.init_mu = float(kwargs.get('init_mu', 0.01))

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        # user CF factors
        self.theta = self.init_std * \
            np.random.randn(self.n_components, n_users).astype(floatX)
        # item CF factors
        self.beta = self.init_std * \
            np.random.randn(self.n_components, n_items).astype(floatX)
        # user exposure factors
        self.phi = self.init_std * \
            np.random.randn(self.n_components, n_users).astype(floatX)
        # user exposure bias
        self.alpha = np.log(self.init_mu / (1 - self.init_mu)) * \
            np.ones((n_users, 1), dtype=floatX)
        # the matrix A
        self.A = self.init_std * \
            np.random.randn(n_users, n_items).astype(floatX)
        # the location exposure mu
        self.mu = self.init_std * \
            np.random.random_sample((n_users, n_items)).astype(floatX)

    def _set_params(self, X, para_dir):
        params = np.load(para_dir)
        self.theta, self.beta, self.phi, self.alpha = params['U'].T, params['V'].T, params['nu'].T, params['alpha']
        self.mu = get_mu(self.phi.T, X, self.alpha)
        self.A = np.random.binomial(1, self.mu, self.mu.shape)

    def fit(self, Y, X, para_dir = None):
        '''Fit the model to the data in X.

        Parameters
        ----------
        Y : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        X : array-like, shape (n_items, n_components)
            item content representations (e.g. topics, locations)

        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Validation data.

        **kwargs: dict
            Additional keywords to evaluation function call on validation data

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_items = Y.shape
        assert X.shape[0] == n_items
        if para_dir is None:
            self._init_params(n_users, n_items)
        else:
            self._set_params(X, para_dir)
        self._update(Y, X)
        return self

    def _predict(self):
        '''calculate the predictive distribution based on current parameters'''
        # make prediction by matrix factorization
        predict1 = self.theta.T.dot(self.beta)
        # make prediction by integrating out the uncertainty from the exposure latent variables
        predict2 = self.mu * predict1 #element-wise multiplication
        return predict1, predict2

    def _MSE(self, target, p1, p2):
        e1 = mean_squared_error(target, p1)
        e2 = mean_squared_error(target, p2)
        return e1, e2
        
    def _update(self, X, pi):
        '''Model training '''
        XT = X.T.tocsr()  # pre-compute this
        if self.verbose:
            print('Start to sample...')
        predict1, predict2 = self._predict()
        U = self.theta
        V = self.beta
        for i in xrange(self.max_iter):
            if self.verbose:
                print('Iteration: #%d' % i)
            self._gibbs_sampler(X, XT, pi)
            p1, p2 = self._predict()
            predict1 = predict1 + p1
            predict2 = predict2 + p2
            U = U + self.theta
            V = V + self.beta
            #self.mse[i, 0], self.mse[i, 1] = self._MSE(X.toarray(), predict1/(i+2), predict2/(i+2))
            self.mse[i, 0], self.mse[i, 1] = self._MSE(X.toarray(), p1, p2)
            if self.verbose:
                print('The MSE are:%0.2f and %0.2f' % (self.mse[i, 0], self.mse[i, 1]))

        num = self.max_iter+1
        self.theta = U / num
        self.beta = V / num

        self._save_params(self.max_iter)
        return predict1 / num, predict2 / num


    def _gibbs_sampler(self, Y, YT, pi):
        '''sample latent variables'''
        #pdb.set_trace()
        U, I = Y.shape  # u = number of users, i = number of items

        #==========================================================================
        #sample the matrix A
        if self.verbose:
            start_t = _writeline_and_time('\tSampling exposure covariate...')

        P = (1.0/self.mu - 1) / scipy.stats.norm(self.theta.T.dot(self.beta), self.lam_y * ones(Y.shape)).pdf(Y.toarray())            
        P = 1 / (1 + P)
        P[np.where(Y.toarray()==1)] = 1
        self.A = np.random.binomial(1, P, self.A.shape)

        if self.verbose:
            print('\r\tSampling exposure covariate: time=%.2f'
                  % (time.time() - start_t))
            
        getGibbs(self,U,I,Y)
        
        #Update user exposure factors and bias with mini-batch SGD
        self._update_expo(YT, pi)


    def _update_expo(self, XT, pi):
        '''Update user exposure factors and bias with mini-batch SGD'''
        #pdb.set_trace()
        if self.verbose:
            start_t = _writeline_and_time(
                '\tUpdating user exposure factors...\n')
        nu_old = self.phi.T.copy()
        alpha_old = self.alpha.copy()

        n_items = XT.shape[0]
        start_idx = range(0, n_items, self.batch_sgd)
        end_idx = start_idx[1:] + [n_items]

        # run SGD to learn phi and alpha
        for epoch in xrange(self.max_epoch):
            idx = np.random.permutation(n_items)

            for lo, hi in zip(start_idx[:-1], end_idx[:-1]):
                A_batch = a_row_batch(XT[idx[lo:hi]], self.beta.T[idx[lo:hi]],
                                      self.theta.T, pi[idx[lo:hi]], nu_old,
                                      alpha_old.T, self.lam_y).T
                pred_batch = get_mu(self.phi.T, pi[idx[lo:hi]], self.alpha)
                diff = A_batch - pred_batch
                grad_nu = 1. / self.batch_sgd * diff.dot(pi[idx[lo:hi]])\
                    - self.lam_nu * self.phi.T
                self.phi += self.learning_rate * grad_nu.T
                self.alpha += self.learning_rate * diff.mean(axis=1,
                                                             keepdims=True)
        if self.verbose:
            print('\tUpdating user exposure factors: time=%.2f'
                  % (time.time() - start_t))
            sys.stdout.flush()
        pass

    def _save_params(self, iter):
        '''Save the parameters'''
        #pdb.set_trace()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'ExpoMF_cov_K%d_mu%.1e_iter%d.npz' % (self.n_components,
                                                         self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta.T,
                 V=self.beta.T, nu=self.phi.T, alpha=self.alpha)
# Utility functions #

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def inv_logit(x):
    return 1. / (1 + np.exp(-x))

def get_mu(nu, pi, alpha):
    ''' \mu_{ui} = inv_logit(nu_u * pi_i + alpha_u)'''
    return inv_logit(nu.dot(pi.T) + alpha)

def a_row_batch(Y_batch, theta_batch, beta, nu_batch, pi, alpha_batch, lam_y):
    '''Compute the posterior of exposure latent variables A by batch

    When updating users:
        Y_batch: (batch_users, n_items)
        theta_batch: (batch_users, n_components)
        beta: (n_items, n_components)
        nu_batch: (batch_users, n_components)
        pi: (n_items, n_components)
        alpha_batch: (batch_users, 1)

    When updating items:
        Y_batch: (batch_item, n_users)
        theta_batch: (batch_item, n_components)
        beta: (n_users, n_components)
        nu_batch: (batch_item, n_components)
        pi: (n_users, n_components)
        alpha_batch: (1, n_users)
    '''
    #pdb.set_trace()
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    mu = get_mu(nu_batch, pi, alpha_batch)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A

def chunkIndex(end, chunksize):
    ''' returns chunked interval 10,3 returns [0,3][3,6][6,9][9,10]'''
    list = []
    i = 0
    while i < end:
        newend = min(end,i+chunksize)
        list.append((i, newend))
        i = newend
    return list

def sampleUser(ev,Y,a,b,k,lB):
    r = range(a,b)
    theta = np.zeros((ev.theta.shape[0],b-a))
    for u in r: # sample for each item
        A_star = sparse.csr_matrix(np.diag(ev.A[u, :]))
        cov = ev.lam_y * ev.beta.dot(A_star.dot(ev.beta.T)) + k
        mean = ev.lam_y * np.linalg.lstsq(cov, ev.beta.dot(A_star.dot(Y[u, :].T.toarray())))[0]
        theta[:,u-a] = np.random.multivariate_normal(mean.flatten(), cov)
    return theta

def sampleItem(ev,Y,a,b,k,lB):
    r = range(a,b)
    beta = np.zeros((ev.beta.shape[0],b-a))
    for i in r: # sample for each item
        A_star = sparse.csr_matrix(np.diag(ev.A[:, i])) 
        cov = lB.dot(A_star.dot(ev.theta.T)) + k
        mean = ev.lam_y * np.linalg.lstsq(cov, ev.theta.dot(A_star.dot(Y[:, i].toarray())))[0]
        beta[:,i-a] = np.random.multivariate_normal(mean.flatten(), cov)
    return beta

def getGibbs(ev,U,I,Y):
    #==========================================================================
    #sample user factors
    if ev.verbose:
        start_t = _writeline_and_time('\tSampling user factors...')

    k = ev.lam_theta * np.eye(ev.n_components) + EPS
    lB = ev.lam_y * ev.theta
    chunkSize = U / max(ev.n_jobs, ev.min_chunk_count)

    thetas = Parallel(ev.n_jobs)(delayed(sampleUser)(ev,Y,i[0],i[1],k,lB) for i in chunkIndex(U, chunkSize))
    
    ci = 0
    for i in chunkIndex(U, chunkSize):
        a,b = i
        r = range(a,b)
        ev.theta[:,a:b] = thetas[ci][:,:]
        ci += 1

    #==========================================================================
    #sample item factors
    if ev.verbose:
        print('\r\tSampling user factors: time=%.2f'
              % (time.time() - start_t))
        start_t = _writeline_and_time('\tSampling item factors...')

    k = ev.lam_beta * np.eye(ev.n_components) + EPS
    lB = ev.lam_y * ev.theta
    chunkSize = I / max(ev.n_jobs, ev.min_chunk_count)

    betas = Parallel(ev.n_jobs)(delayed(sampleItem)(ev,Y,i[0],i[1],k,lB) for i in chunkIndex(I, chunkSize))
    
    ci = 0
    for i in chunkIndex(I, chunkSize):
        a,b = i
        r = range(a,b)
        ev.beta[:,a:b] = betas[ci][:,:]
        ci += 1
        
    if ev.verbose:
        print('\r\tSampling item factors: time=%.2f'
              % (time.time() - start_t))

