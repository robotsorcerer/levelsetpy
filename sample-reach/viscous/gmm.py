""" This file defines a Gaussian mixture model class. """
import logging

import torch 
import numpy as np
import scipy.linalg


LOGGER = logging.getLogger(__name__)

def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = torch.max(vec, dim=axis, keepdim=keepdims)
    maxv[maxv == -float('inf')] = 0
    return torch.log(torch.sum(torch.exp(vec-maxv), dim=axis, keepdim=keepdims)) + maxv

def check_sigma(A):
    """
        checks if the sigma matrix is symmetric
        positive definite before inverting via cholesky decomposition
    """
    eigval = torch.linalg.eigh(A)[0]
    if torch.equal(A, A.T) and torch.all(eigval>0):
        # LOGGER.debug("sigma is pos. def. Computing cholesky factorization")
        return A
    else:
        eta = 1e-6  # Regularizer for matrix multiplier
        low = torch.min(eigval)
        Anew = low * A + eta * torch.eye(A.shape[0], device=A.device)
        return Anew

class GMM:
    """ Gaussian Mixture Model. """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None
        
    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            pts: A N x D tensor of points.
        """
        logwts = self.clusterwts(pts)
        mu0, Phi = self.moments(logwts)
        m = self.N
        n0 = m - 2 - mu0.shape[0]
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        """
        logobs = self.estep(data)
        logwts = logobs - torch.logsumexp(logobs, dim=1, keepdim=True)
        logwts = torch.logsumexp(logwts, dim=0) - torch.log(torch.tensor(data.shape[0]))
        return logwts.T

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        """
        N, D = data.shape
        K = self.sigma.shape[0]
        logobs = -0.5 * torch.ones((N, K), device=data.device) * D * torch.log(torch.tensor(2 * torch.pi))
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = torch.linalg.cholesky(sigma)
            logobs[:, i] -= torch.sum(torch.log(torch.diag(L)))
            diff = (data - mu).T
            soln = torch.linalg.solve_triangular(L, diff, upper=False)
            logobs[:, i] -= 0.5 * torch.sum(soln ** 2, dim=0)
        logobs += self.logmass.T
        return logobs

    def moments(self, logwts):
        """
        Compute the moments of the cluster mixture with logwts.
        """
        wts = torch.exp(logwts)
        mu = torch.sum(self.mu * wts, dim=0)
        diff = self.mu - mu.unsqueeze(0)
        diff_expand = self.mu.unsqueeze(1) * diff.unsqueeze(2)
        wts_expand = wts.unsqueeze(2)
        sigma = torch.sum((self.sigma + diff_expand) * wts_expand, dim=0)
        return mu, sigma
