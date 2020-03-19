from functools import reduce
from random import random, normalvariate

import numpy as np
from matplotlib import pylab as plt
from scipy.stats import randint, binom, geom, poisson, logser, stats

import distributionClass as dist
import pandas as pd


class UniformDiscreteDistribution(dist.DiscreteDistribution):
    def __init__(self, r_low, r_up, n):
        super().__init__(r_low, r_up)
        self._sample = pd.Series(
            [np.math.floor((r_up - r_low + 1) * np.random.random_sample() + r_low) for i in range(n)])

    def theoretical_probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, randint.pmf(x, self.r_low, self.r_up + 1), 'bo')

    def theoretical_distribution(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        y = randint.cdf(x, self.r_low, self.r_up + 1)
        plt.plot(x, y, 'ro')

    def kolmogorov_criterion(self):
        n = len(self.sample)
        x, emp_cdf = self.distribution_function()
        cdf = randint.cdf(x, self.r_low, self.r_up + 1)
        d = emp_cdf - cdf
        d_n = np.amax(np.absolute(d))
        s_k = d_n * np.sqrt(n)
        print('uniform', s_k)


class BinomialDistribution(dist.DiscreteDistribution):
    def __init__(self, p, countExp, n):
        super().__init__(0, countExp)
        self._p = p
        if countExp < 10:
            sample = []
            p0 = (1 - p) ** countExp
            for i in range(n):
                px = p0
                rand = np.random.random_sample()
                for x in range(countExp):
                    rand -= px
                    if rand < 0:
                        sample.append(x)
                        break
                    px = px * (countExp - x) * p / ((x + 1) * (1 - p))
            self._sample = pd.Series(sample)
        else:
            self._sample = pd.Series([np.math.floor(normalvariate(countExp * p, np.sqrt(countExp * p * (1.0 - p)))
                                                    + 0.5) for _ in range(n)])

    def theoretical_probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, binom.pmf(x, self.r_up, self.p), 'bo')

    def theoretical_distribution(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, binom.cdf(x, self.r_up, self.p), 'ro')

    def kolmogorov_criterion(self):
        n = len(self.sample)
        x, emp_cdf = self.distribution_function()
        cdf = binom.cdf(x, self.r_up, self.p)
        d = emp_cdf - cdf
        d_n = np.amax(np.absolute(d))
        s_k = d_n * np.sqrt(n)
        print('binom', s_k)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value


class GeometricDistribution(dist.DiscreteDistribution):
    def __init__(self, p, n, method='cumulative'):
        super().__init__(1, 1)
        self._p = p
        sample = []
        if method == 'cumulative':
            p0 = self.p
            for i in range(1, n):
                px = p0
                rand = np.random.random_sample()
                x = 1
                while True:
                    rand -= px
                    if rand < 0:
                        sample.append(x)
                        self.update_r_up(x)
                        break
                    px = px * (1 - p)
                    x += 1
        elif method == 'direct':
            for i in range(1, n):
                x = 1
                while np.random.random_sample() > p:
                    x += 1
                sample.append(x)
                self.update_r_up(x)
        elif method == 'upgrade cumulative':
            for i in range(1, n):
                rand = np.random.random_sample()
                x = int(np.log(rand) / np.log(1 - self.p)) + 1
                sample.append(x)
                self.update_r_up(x)

        self._sample = pd.Series(sample)

    def kolmogorov_criterion(self):
        n = len(self.sample)
        x, emp_cdf = self.distribution_function()
        cdf = geom.cdf(x, self.p)
        d = emp_cdf - cdf
        d_n = np.amax(np.absolute(d))
        s_k = d_n * np.sqrt(n)
        print('geometric', s_k)

    def theoretical_probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, geom.pmf(x, self.p), 'bo')

    def theoretical_distribution(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, geom.cdf(x, self.p), 'ro')

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value


class PoissonDistribution(dist.DiscreteDistribution):
    def __init__(self, lamb, n, method='cumulative'):
        super().__init__(0, 1)
        self._lamb = lamb
        sample = []
        if method == 'cumulative':
            p0 = np.exp(- self.lamb)
            for i in range(1, n):
                px = p0
                rand = np.random.random_sample()
                x = 0
                while True:
                    rand -= px
                    if rand < 0:
                        sample.append(x)
                        self.update_r_up(x)
                        break
                    x += 1
                    px = px * self.lamb / x
        elif method == 'direct':
            for i in range(0, n):
                rand = np.random.random_sample()
                x = 0
                while True:
                    rand = rand * np.random.random_sample()
                    x += 1
                    if rand < np.exp(- self.lamb):
                        sample.append(x)
                        self.update_r_up(x)
                        break

        elif method == 'approximate':
            for i in range(n):
                x = np.math.floor(normalvariate(self.lamb, self.lamb))
                sample.append(x)
                self.update_r_up(x)

        self._sample = pd.Series(sample)

    def theoretical_probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, poisson.pmf(x, self.lamb), 'bo')

    def theoretical_distribution(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, poisson.cdf(x, self.lamb), 'ro')

    def kolmogorov_criterion(self):
        n = len(self.sample)
        x, emp_cdf = self.distribution_function()
        cdf = poisson.cdf(x, self.lamb)
        d = emp_cdf - cdf
        d_n = np.amax(np.absolute(d))
        s_k = d_n * np.sqrt(n)
        print('poisson', s_k)

    @property
    def lamb(self):
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        self._lamb = value


class LogarithmicDistribution(dist.DiscreteDistribution):
    def __init__(self, p, n, method='cumulative'):
        super().__init__(1, 1)
        self._p = p
        sample = []
        if method == 'cumulative':
            p0 = - self.p / np.log(1 - self.p)
            for i in range(1, n):
                px = p0
                rand = np.random.random_sample()
                x = 1
                while True:
                    rand -= px
                    if rand < 0:
                        sample.append(x)
                        self.update_r_up(x)
                        break
                    px = px * x * self.p / (x + 1)
                    x += 1

        self._sample = pd.Series(sample)

    def theoretical_probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, logser.pmf(x, self.p), 'bo')

    def theoretical_distribution(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        plt.plot(x, logser.cdf(x, self.p), 'ro')

    def kolmogorov_criterion(self):
        n = len(self.sample)
        x, emp_cdf = self.distribution_function()
        cdf = logser.cdf(x, self.p)
        d = emp_cdf - cdf
        d_n = np.amax(np.absolute(d))
        s_k = d_n * np.sqrt(n)
        print('logser', s_k)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value
