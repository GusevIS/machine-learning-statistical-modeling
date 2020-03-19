from abc import ABC

import numpy as np
from matplotlib import pylab as plt
import seaborn as sns
import abc
import pandas as pd
import seaborn as sns


class Distribution(abc.ABC):
    def __init__(self, r_low, r_up):
        self._r_low = r_low
        self._r_up = r_up
        self._sample = None

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def r_low(self):
        return self._r_low

    @r_low.setter
    def r_low(self, value):
        self._r_low = value

    @property
    def r_up(self):
        return self._r_up

    @r_up.setter
    def r_up(self, value):
        self._r_up = value

    def __str__(self):
        return '{u._sample}'.format(u=self)

    def hist(self):
        self.sample.hist(bins=10, density=True, rwidth=1, alpha=0.8)

    def distribution_function(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        n = len(self.sample)
        sorted_sample = np.sort(self.sample)
        return x, np.array([len(list(filter(lambda xi: xi <= i, sorted_sample))) / n for i in x])

    def plt_distribution_function(self):
        x, y = self.distribution_function()
        plt.ylim(0.0, 1.1)
        plt.plot(x, y, 'bo')
        plt.vlines(x, 0, ymax=y, colors='b', lw=1, alpha=0.5)

    def graphic_test(self):
        x = list(self.sample[:-1])
        y = list(self.sample[1:])
        n = len(x)
        """""
        #plt.plot(x, y, 'bo')
        xy = pd.DataFrame({'x': x, 'y': y})
        print(xy)
        #dfcounts = xy.groupby(['x', 'y']).size().reset_index(name='counts')
        color = [range(len(x))]
        xy.plot.scatter(x="x", y="y", colormap='viridis', c=color)
        """""
        with sns.axes_style("white"):
            sns.jointplot(x=x, y=y, kind="kde", color="k")

    def mean(self):
        return np.mean(self.sample)

    def dispersion(self):
        return np.var(self.sample)

    def update_r_up(self, x):
        if self.r_up <= x:
            self.r_up = x

    @abc.abstractmethod
    def theoretical_probability(self):
        pass

    @abc.abstractmethod
    def probability(self):
        pass


class DiscreteDistribution(Distribution, ABC):
    def probability(self):
        x = np.array([i for i in range(self.r_low, self.r_up + 1)])
        n = len(self.sample)
        y = np.array([len(list(filter(lambda xi: xi == i, self.sample))) / n for i in x])
        plt.ylim(0.0, 1.0)
        plt.plot(x, y, 'ro')
        plt.vlines(x, 0, ymax=y, colors='r', lw=1, alpha=0.5)
