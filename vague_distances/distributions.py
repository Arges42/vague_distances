"""Distributions to use for the initialization of VagueAreas and VagueIntervals.

"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import truncnorm


class Distribution(ABC):
    """Abstract base class for all distributions.
    """

    def __init__(self, discrete=False, random_state=None):
        if not random_state is None:
            np.random.seed(random_state)

        self.discrete = discrete

    @abstractmethod
    def sample(self, n_samples=1):
        """Sample from the distribution.

        Attributes:
            n_samples -- Nummber of samples
        """
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class UniformDistribution2D(Distribution):
    """Two dimensional unfirom distribution.
    """

    def __init__(self, x1, x2, y1, y2, **kwargs):
        """
        Attributes:
            x1 -- lower x value
            x2 -- upper x value
            y1 -- lower y value
            y2 -- upper y value
            random_state -- Pass a random state to numpy
        """

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self._prob = 1./((x2-x1)*(y2-y1))

        super().__init__(kwargs)

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Attributes:
            n_samples -- Nummber of samples
        """
        
        if self.discrete:
            x = np.random.randint(self.x1, self.x2, n_samples)
            y = np.random.randint(self.y1, self.y2, n_samples)
        else:
            x = (self.x2 - self.x1) * np.random.random_sample(n_samples) + self.x1
            y = (self.y2 - self.y1) * np.random.random_sample(n_samples) + self.y1
        return np.stack((x, y), axis=-1)

    def __call__(self, x, y):
        cond_x = np.logical_and(self.x1 <= x, x <= self.x2)
        cond_y = np.logical_and(self.y1 <= y, y <= self.y2)
        return np.where(np.logical_and(cond_x, cond_y), self._prob, 0)


class UniformDistribution(Distribution):
    """One dimensional uniform distribution."""

    def __init__(self, a, b, **kwargs):
        """
        Attributes:
            a -- Lower interval value
            b -- Upper interval value
            random_state -- Pass a random state to numpy
        """
        self.a = a
        self.b = b
        self._prob = 1./(b-a)

        super().__init__(kwargs)

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Attributes:
            n_samples -- Nummber of samples
        """
        if self.discrete:
            return np.random.randint(self.a, self.b, n_samples)
        else:
            return (self.b - self.a) * np.random.random_sample(n_samples) + self.a

    def __call__(self, x):
        return np.where(np.logical_and(self.a <= x, x <= self.b), self._prob, 0)


class TruncatedGaussian(Distribution):
    """Wrapper class for simplified calls to scipy.stats.trucnorm

    """

    def __init__(self, a, b, mu, sig, random_state=None):
        self.mu = mu
        self.sig = sig

        self.lower = (a - self.mu)/self.sig
        self.upper = (b - self.mu)/self.sig

        super().__init__(random_state)

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Attributes:
            n_samples -- Nummber of samples
        """
        return truncnorm.rvs(self.lower, self.upper, self.mu, self.sig, n_samples)

    def __call__(self, x):
        return truncnorm.pdf(x, self.lower, self.upper, self.mu, self.sig)
