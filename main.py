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

plt.show()
