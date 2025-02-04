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
col['smallsamplemeans'].hist(bins=500, density=True)
col['largesamplemeans'].hist(bins=500, density=True)
# histogram will not look normal for small samples, but with large sample it will look normal

plt.show()
