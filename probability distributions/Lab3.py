from continuousDistribution import UniformContinuousDistribution, NormalContinuousDistribution, \
    ExponentialContinuousDistribution, ChisquareContinuousDistribution, StudentContinuousDistribution
from matplotlib import pylab as plt
"""""
uniform = UniformContinuousDistribution(10, 23, 10000)
uniform.pdf()
uniform.theoretical_pdf()
plt.figure('distribution f')
uniform.plt_distribution_function()
uniform.theoretical_distribution()
"""""
"""""
normSample = NormalContinuousDistribution(0, 1, 10000)
normSample.theoretical_pdf()
normSample.pdf()
plt.figure('distribution f')
normSample.plt_distribution_function()
normSample.theoretical_distribution()
"""
"""
expSample = ExponentialContinuousDistribution(1, 10000)
expSample.theoretical_pdf()
expSample.pdf()
plt.figure('distribution f')
expSample.plt_distribution_function()
expSample.theoretical_distribution()
print(expSample.mean())
print(expSample.dispersion())
"""
"""
chisquareSample = ChisquareContinuousDistribution(5, 10000)
chisquareSample.pdf()
chisquareSample.theoretical_pdf()

plt.figure('distribution f')
chisquareSample.plt_distribution_function()
chisquareSample.theoretical_distribution()
"""
studentSample = StudentContinuousDistribution(10, 10000)
studentSample.pdf()
studentSample.theoretical_pdf()
plt.figure('distribution f')
studentSample.plt_distribution_function()
studentSample.theoretical_distribution()

plt.show()
