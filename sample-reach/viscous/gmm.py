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
    """ 
        Expecttaion-maximization operator for the normal inverse Wishart surrogate used to compute the 
        proximal operator for the spatial graidents of the value function.
    """

    def __init__(self, device='cpu', init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None
        self.device = device
        
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
            Compute cluster weights for specified points under the Gaussian distribution.
        """

        logobs = self.estep(data)
        logwts = logobs - torch.logsumexp(logobs, dim=1, keepdim=True)
        logwts = torch.logsumexp(logwts, dim=0) - torch.log(torch.tensor(data.shape[0]))

        return logwts.T

    def estep(self, data):
        """
            Compute log observation probabilities under expectation maximization.
        """

        N, D = data.shape[:2]
        K = self.sigma.shape[0]
        logobs = -0.5 * torch.ones_like((N, K), device=self.device) * D * torch.log(torch.tensor(2 * torch.pi))
        # logobs = -0.5 * torch.ones_like(data, device=self.device) * D * torch.log(torch.tensor(2 * torch.pi))

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
        Compute the moments of the cluster with logwts.
        """
        wts = torch.exp(logwts)
        mu = torch.sum(self.mu * wts, dim=0)
        diff = self.mu - mu.unsqueeze(0)
        diff_expand = self.mu.unsqueeze(1) * diff.unsqueeze(2)
        wts_expand = wts.unsqueeze(2)
        sigma = torch.sum((self.sigma + diff_expand) * wts_expand, dim=0)

        return mu, sigma
    
    def update(self, data, K, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        LOGGER.debug('Fitting GMM with %d clusters on %d points.', K, N)

        if (not self.warmstart or self.sigma is None or
                K != self.sigma.shape[0]):
            # Initialization.
            LOGGER.debug('Initializing GMM.')
            self.sigma = torch.zeros((K, Do, Do), device=data.device)
            self.mu = torch.zeros((K, Do), device=data.device)
            self.logmass = torch.log(torch.ones(K, 1, device=data.device) / K)
            self.mass = torch.ones(K, 1, device=data.device) / K
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = torch.randint(0, K, (1, N), device=data.device)
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i).squeeze()
                mu = torch.mean(data[cluster_idx, :], dim=0)
                diff = (data[cluster_idx, :] - mu).T
                sigma = (1.0 / K) * torch.matmul(diff, diff.T)
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + torch.eye(Do, device=data.device) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = torch.sum(torch.logsumexp(logobs, dim=1))
            LOGGER.debug('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)
            if ll < prevll:
                LOGGER.debug('Log-likelihood decreased! Ending on itr=%d/%d',
                             itr, max_iterations)
                break
            if torch.abs(ll - prevll) < 1e-5 * prevll:
                LOGGER.debug('GMM converged on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - torch.logsumexp(logobs, dim=1, keepdim=True)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - torch.logsumexp(logw, dim=0, keepdim=True)
            assert logwn.shape == (N, K)
            w = torch.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = torch.sum(logw, dim=0).view(-1, 1)
            self.logmass = self.logmass - torch.logsumexp(self.logmass, dim=0, keepdim=True)
            assert self.logmass.shape == (K, 1)
            self.mass = torch.exp(self.logmass)

            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4).squeeze()] = 1.0 / N
            # Fit cluster means.
            w_expand = w.unsqueeze(2)
            data_expand = data.unsqueeze(1)
            self.mu = torch.sum(w_expand * data_expand, dim=0)
            # Fit covariances.
            wdata = data_expand * torch.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = torch.matmul(wdata[:, i, :].T, wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - torch.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * torch.eye(Do, device=data.device)
