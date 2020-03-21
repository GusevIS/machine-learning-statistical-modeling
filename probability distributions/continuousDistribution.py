import math

import numpy as np
from matplotlib import pylab as plt
from scipy.special._ufuncs import erf, gamma, gammainc, hyp2f1
from scipy.stats import uniform

import distributionClass as dist
import pandas as pd


class UniformContinuousDistribution(dist.ContinuousDistribution):
    def __init__(self, r_low, r_up, n):
        super().__init__(r_low, r_up)
        self.sample = pd.Series([(self.r_up - self.r_low) * np.random.random_sample() + self.r_low for _ in range(n)])

    def theoretical_pdf(self):
        x = np.linspace(self.r_low, self.r_up)
        y = np.array([1 / (self.r_up - self.r_low) for _ in x])
        plt.plot(x, y, 'r-', lw=3)
        self.sample.hist(density=True, alpha=0.4, width=1.1)

    def theoretical_distribution(self):
        x = np.sort(self.sample)
        y = np.array([(i - self.r_low) / (self.r_up - self.r_low) for i in x])
        plt.plot(x, y, 'r-')


class NormalContinuousDistribution(dist.ContinuousDistribution):
    def __init__(self, mean, std, n, method='CLT'):
        super().__init__(0, 0)
        self._mean = mean
        self._std = std
        sample = []
        if method == 'CLT':
            count = 12
            for i in range(n):
                uniform_sample = np.random.random_sample(count)
                rand = mean + std * (np.sum(uniform_sample) - 0.5 * count) * (6 / np.sqrt(3 * count))
                sample.append(rand)

        elif method == 'Box-Miller':
            assert (self.mean == 0 and self.std == 1), \
                'Box-Miller can generate only standart norm distribution with mean = 0, std = 1'
            for i in range(math.ceil((float(n) / 2))):
                u1 = np.random.random_sample(1)
                u2 = np.random.random_sample(1)
                rand1 = np.sqrt(- 2 * np.log(u2)) * np.cos(2 * np.pi * u1)
                rand2 = np.sqrt(- 2 * np.log(u2)) * np.sin(2 * np.pi * u1)
                sample.append(rand1[0])
                sample.append(rand2[0])
            sample = np.around(sample, decimals=6)

        self.sample = pd.Series(sample)
        self.r_low = np.min(self.sample)
        self.r_up = np.max(self.sample)

    def theoretical_pdf(self):
        x = np.linspace(self.r_low, self.r_up, 100)
        y = np.array([1 * np.exp(- ((i - self.mean) ** 2) / (2 * self.std ** 2)) / (self.std * np.sqrt(2 * np.pi))
                      for i in x])
        plt.plot(x, y, 'r-', lw=3)
        self.sample.hist(density=True, alpha=0.4)

    def theoretical_distribution(self):
        x = np.sort(self.sample)
        y = np.array([0.5 * (1 + erf((i - self.mean) / np.sqrt(2 * self.std ** 2))) for i in x])
        plt.plot(x, y, 'r-')

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value):
        self._std = value


class ExponentialContinuousDistribution(dist.ContinuousDistribution):
    def __init__(self, beta, n):
        super().__init__(0, 0)
        self._beta = beta
        sample = []
        for i in range(n):
            rand = - self.beta * np.log(np.random.random())
            sample.append(rand)

        self.sample = pd.Series(sample)
        self.r_low = np.min(self.sample)
        self.r_up = np.max(self.sample)

    def theoretical_pdf(self):
        x = np.linspace(self.r_low, self.r_up, 100)
        y = np.array([np.exp(- i / self.beta) / self.beta for i in x])
        plt.plot(x, y, 'r-', lw=3)
        self.sample.hist(density=True, alpha=0.4)

    def theoretical_distribution(self):
        x = np.sort(self.sample)
        y = np.array([1 - np.exp(- i / self.beta) for i in x])
        plt.plot(x, y, 'r-')

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self.beta = value


class ChisquareContinuousDistribution(dist.ContinuousDistribution):
    def __init__(self, freedom_degrees, n):
        super().__init__(0, 0)
        self._freedom_degrees = freedom_degrees
        sample = []
        for i in range(n):
            rand = np.sum(NormalContinuousDistribution(0, 1, self.freedom_degrees).sample ** 2)
            sample.append(rand)

        self.sample = pd.Series(sample)
        self.r_low = np.min(self.sample)
        self.r_up = np.max(self.sample)

    def theoretical_pdf(self):
        gam = gamma(self.freedom_degrees / 2)
        x = np.linspace(self.r_low, self.r_up, 100)
        y = np.array([((i / 2) ** ((self.freedom_degrees / 2) - 1)) * np.exp(- i / 2) / (2 * gam) for i in x])
        plt.plot(x, y, 'r-', lw=3)
        self.sample.hist(density=True, alpha=0.4)

    def theoretical_distribution(self):
        x = np.sort(self.sample)
        y = np.array([gammainc(self.freedom_degrees / 2, i / 2) for i in x])
        plt.plot(x, y, 'r-')

    @property
    def freedom_degrees(self):
        return self._freedom_degrees

    @freedom_degrees.setter
    def freedom_degrees(self, value):
        self.freedom_degrees = value


class StudentContinuousDistribution(dist.ContinuousDistribution):
    def __init__(self, freedom_degrees, n):
        super().__init__(0, 0)
        self._freedom_degrees = freedom_degrees
        sample = []
        for i in range(n):
            rand = NormalContinuousDistribution(0, 1, 1).sample[0] / np.sqrt(
                ChisquareContinuousDistribution(self.freedom_degrees, 1).sample[0] / self.freedom_degrees)
            sample.append(rand)

        self.sample = pd.Series(sample)
        self.r_low = np.min(self.sample)
        self.r_up = np.max(self.sample)

    def theoretical_pdf(self):
        gam = gamma(self.freedom_degrees / 2)
        gam2 = gamma((self.freedom_degrees + 1) / 2)
        x = np.linspace(self.r_low, self.r_up, 100)
        y = np.array([gam2 * ((1 + i ** 2 / self.freedom_degrees) ** (-(self.freedom_degrees + 1) / 2)) /
                      (np.sqrt(self.freedom_degrees * np.pi) * gam) for i in x])
        plt.plot(x, y, 'r-', lw=3)
        self.sample.hist(density=True, alpha=0.4)

    def theoretical_distribution(self):
        x = np.sort(self.sample)
        gam = gamma(self.freedom_degrees / 2)
        gam2 = gamma((self.freedom_degrees + 1) / 2)
        y = np.array([0.5 + i * gam2 * (hyp2f1(0.5, (self.freedom_degrees + 1) / 2, 1.5, - (i ** 2) / self.freedom_degrees)) / (np.sqrt(np.pi * self.freedom_degrees) * gam) for i in x])
        plt.plot(x, y, 'r-')

    @property
    def freedom_degrees(self):
        return self._freedom_degrees

    @freedom_degrees.setter
    def freedom_degrees(self, value):
        self.freedom_degrees = value
