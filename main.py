import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Population - group of individuals with common property
# Sample - small randomly selected group of population
# Random sampling - with or without replacement (after selecting individual indclude this one for next selection?)

data = pd.DataFrame()
data['population'] = [47, 48, 85, 20, 19, 13, 72, 16, 50, 60]

sample_no_replacement = data['population'].sample(5, replace=False)  # replace=False by default
sample_with_replacement = data['population'].sample(5, replace=True)
print(sample_no_replacement)  # never the same number chosen
print(sample_with_replacement)  # sometimes numbers will repeat

# Characteristics like mean, variance etc. are named as parameters for populations, but as statitics in samples.
# With fixed population parameters will never change.
print('\nPopulation parameters:'
      '\n              mean: ', data['population'].mean(),
      '\nstandard deviation: ', data['population'].std(ddof=0),
      '\n          variance: ', data['population'].var(ddof=0),
      # ddof sets denominator for variance and standard deviation, ddof=0 means that denominator equals population size
      '\n              size: ', data['population'].shape[0])
print('\nSample statistics:'
      '\n              mean: ', sample_with_replacement.mean(),
      '\nstandard deviation: ', sample_with_replacement.std(ddof=1),
      '\n          variance: ', sample_with_replacement.var(ddof=1),
      # ddof=1 means that denominator equals (sample size - 1)
      # default for ddof is 1
      # why ddof is different for populations and samples? because math says that it is better that way ;)
      '\n              size: ', sample_with_replacement.shape[0])


# Sampling from normal distribution
Fstsample = pd.DataFrame(np.random.normal(10, 5, size=30))
print('\nSample statistics:'
      '\n              mean: ', Fstsample[0].mean(),  # should be around 10
      '\nstandard deviation: ', Fstsample[0].std(ddof=1))  # should be around 5
# Variation of sample - different samples yield different mean and std,
# but because samples are taken from same population there are rules over how mean and std changes


# Empirical distribution of sample mean and variance
meanlist = []
varlist = []
for _ in range(1000):
    sample = pd.DataFrame(np.random.normal(10, 5, size=30))
    meanlist.append(sample[0].mean())
    varlist.append(sample[0].var(ddof=1))

print(meanlist)
print(varlist)

collection = pd.DataFrame()
collection['meanlist'] = meanlist
collection['varlist'] = varlist
# collection['meanlist'].hist(bins=200, density=True)  # normed=1 is not working, probably replaced by density=True
# collection['varlist'].hist(bins=100, density=True)
# meanlist looks somewhat like normal distribution
# varlist is skewed to the right


# Central limit theorem
# If the sample size is larger enough, the distribution of sample mean is approximately normal with N(mu, sigma^2/n)
smallsamplemeanlist = []
largesamplemeanlist = []
pop = pd.DataFrame([1, 0, 1, 0, 1])
for _ in range(100000):
    sample = pop[0].sample(10, replace=True)
    smallsamplemeanlist.append(sample.mean())
    sample = pop[0].sample(2000, replace=True)
    largesamplemeanlist.append(sample.mean())
col = pd.DataFrame()
col['smallsamplemeans'] = smallsamplemeanlist
col['largesamplemeans'] = largesamplemeanlist
# col['smallsamplemeans'].hist(bins=500, density=True)
# col['largesamplemeans'].hist(bins=500, density=True)
# histogram will not look normal for small samples, but with large sample it will look normal
# TODO: I'm not sure if it's true. It'll always look somewhat normal,
#  but with small sample it's more likely to be skewed. And with large sample you have
#  a bell curve multiplied by something like sine wave.


# Confidence interval for daily return
from scipy.stats import norm

data = pd.read_csv('data/TRADEGATE%DGD=D.csv', sep='\t',
                   names=('timestamp', 'open', 'high', 'low', 'close', 'volume', 'is_generated'))
data = data[data['is_generated'] < 1]
data['log_return'] = np.log(data['close']) - np.log(data['close'].shift(1))
# values for calculating the 80% confidence interval
confidence = 0.8
left_bound = (1 - confidence) / 2
right_bound = 1 - left_bound
z_left = norm.ppf(left_bound)
z_right = norm.ppf(right_bound)  # z_right = -z_left for Z-distribution
sample_mean = data['log_return'].mean()
sample_std = data['log_return'].std(ddof=1) / (data.shape[0] ** 0.5)
# assumptions for sample_std:
# - prices are not correlated between each other - false, prices are usually strongly correlated
# - this sample_std math is used when we have sample with replacement,
#   or sample is less than 5% (maybe 10%) of total population
interval_left = sample_mean + z_left * sample_std  # can be converted to norm.ppf(left_bound, sample_mean, sample_std)
interval_right = sample_mean + z_right * sample_std
print('\nSample stats:'
      '\n          mean: ', sample_mean,
      '\n           std: ', sample_std,
      '\n interval left: ', interval_left,
      '\ninterval right: ', interval_right)


# Hypothesis testing
# We can see on the plot that price is in upward trend, but is daily return positive?
# Plots and histogram of daily return gave not obvoius answers.
# First we set a hypotesis - there is Null Hypothesis (H_0; usually assertion we are against)
#   and Alternative Hypothesis (H_a; accepted conclusion after rejecting null).
# Example - H_0: average daily return = 0; H_a: average daily return != 0
# x_bar - sample mean
# mu - population mean
# Assuming that H_0 is correct: mu = 0 => |x_bar - mu| ~= 0 (not large)
# si - population std
# n - sample size
# Standarization (z-distribution): z_hat = (x_bar - mu) / (si / (n ** 0.5))
# si is not known (usually) - using sample std (s) as replacement
# New standarization (t-distribution): t_hat = (x_bar - mu) / (s / (n ** 0.5)) - it's tails are fatter.
# t-distribution is dependent on degree of freedom id sample std (n-1), which increases with increased sample size.
# We can treat t-distribution as z-distribution with large enough sample size. TODO: meaning? 5% population size?
# z_hat = (x_bar - mu) / (s / (n ** 0.5))
x_bar = data['log_return'].mean()
mu = 0  # null assumption
s = data['log_return'].std(ddof=1)
n = data['log_return'].shape[0]
z_hat = (x_bar - mu) / (s / (n ** 0.5))
print(z_hat)

# Decision criteria
# two-tailed test - rejection regions before 2.5% and after 97.5% (alpha = 5%)
# If z_hat is in the rejecion region, then we reject H_0.
# For alpha = 5%, z_hat ~= +-1.96
# Type 1 error - make wrong decision for accepting or rejection H_0
alpha = 0.05
z_left = norm.ppf(alpha/2, 0, 1)
z_right = norm.ppf(1 - alpha/2, 0, 1)  # should equals (-z_left)
print("Should we reject H_0? ", z_left > z_hat or z_hat > z_right)
print("Probability of type 1 error: ", alpha * 100, "%")

# Hypothesis for 1 tail test
# H_0: mu <= 0; H_a > 0
z_right = norm.ppf(1 - alpha, 0, 1)
print("Should we reject H_0? ", z_hat > z_right)
print("Probability of type 1 error: ", alpha * 100, "%")

# P-value - probability of value being more extreme than observation
# Reject H_0 if p < alpha
p = 2 * (1 - norm.cdf(abs(z_hat), 0, 1))
print('Reject H_0? ', p < alpha, ' (Significance: ', alpha * 100, '%)')
# For H1: mu < 0 => p = norm.cdf(z_hat, 0, 1)
# For H2: mu > 0 => p = 1 - norm.cdf(z_hat, 0, 1)


plt.show()
