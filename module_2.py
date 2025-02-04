# rolling 2 6-sided dices
import pandas as pd
import matplotlib.pyplot as plt

# create die
die = pd.DataFrame([1, 2, 3, 4, 5, 6])

# get sample of throwing two dice
dice_throws = die.sample(2, replace=True)
print(dice_throws)
print(dice_throws.sum())
print('Sum of dice:', dice_throws.sum().loc[0])

# get series of throwing two dice
trials = 50
outcomes = pd.DataFrame([die.sample(2, replace=True).sum().loc[0] for _ in range(trials)])
print(outcomes)

# get frequency of provided results
frequency = outcomes[0].value_counts().sort_index()
print(frequency)

# get relative frequency
relative_frequency = frequency / trials  # can also be frequency / frequency.sum()
print(relative_frequency)

# theoretical relative frequency for rolling 2 dices (distribution table)
X_distr = pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # in lesson option index is used
X_distr['prob'] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
X_distr['prob'] /= 36
print(X_distr)
# X_distr['prob'].plot(kind='bar', color='purple')

# get expectation (mean) and variance of randomly distributed variables
mean = (X_distr[0] * X_distr['prob']).sum()
print(mean)
variance = ((X_distr[0] - mean) ** 2 * X_distr['prob']).sum()
print(variance)

# mean and variance of generated results
rolls = 1000
results = pd.Series([die.sample(2, replace=True).sum().loc[0] for _ in range(rolls)])
print(results.mean())
print(results.var())


import numpy as np

data_header = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'is_generated']
data = pd.read_csv('data/TRADEGATE%DGD=D.csv', delimiter='\t', names=data_header)
print(data.tail(10))
# data.loc[2000:, 'close'].plot()

# Log daily return
data['log_daily_return'] = np.log(data['close']) - np.log(data['close'].shift(1))
# data['log_daily_return'].hist(bins=50)


from scipy.stats import norm

# Probability Density Function (PDF)
# Cumulative Distribution Function (CDF)
density = pd.DataFrame()
density['x'] = np.arange(-4, 4, 0.01)
# density['x'].plot()
density['pdf'] = norm.pdf(density['x'], 0, 1)  # norm.pdf(<data>, <mean>, <std>)
# plt.plot(density['x'], density['pdf'])
density['cdf'] = norm.cdf(density['x'], 0, 1)  # norm.cdf(<data>, <mean>, <std>)
# plt.plot(density['x'], density['cdf'])

# get mean and standard deviation form data
mu = data['log_daily_return'].mean()
std = data['log_daily_return'].std(ddof=1)
print(mu)
print(std)

# plot PDF
data['log_daily_return'].hist(bins=50, density=True)
plt_lim = [-0.15, 0.15, 0, 10]  # <xlim left>, <xlim right>, <ylim left>, <ylim right>
data_density = pd.DataFrame()
data_density['x'] = np.arange(plt_lim[0], plt_lim[1] + 0.01, 0.002)
data_density['pdf'] = norm.pdf(data_density['x'], mu, std)
plt.plot(data_density['x'], data_density['pdf'])
print(data_density)

# chance daily loss >5%
data_density_reduced = data_density[round(data_density['x'], 3) <= -0.05]
# for some reason values generated using np.arange have not exact values,
# so they need to be rounded before using comparisons
plt.fill_between(x=data_density_reduced['x'], y1=0, y2=data_density_reduced['pdf'], facecolor='pink', alpha=0.5)
# TODO: code in 3:13 looks strange, but is my code better?
print(norm.cdf(-0.05, mu, std))
data_density['cdf'] = norm.cdf(data_density['x'], mu, std)
# plt.plot(data_density['x'], data_density['cdf'])


# Probability of price dropping 40% in 220 days
# assumption - daily returns are independent
# Sum of independent normal random variables
# mean_sum = mean_1 + mean_2 + ... + mean_n
# std_sum^2 = std_1^2 + std_2^2 + ... + std_n^2  # std^2 is called variance
mean220 = mu * 220
std220 = (220 ** 0.5) * std
print(norm.cdf(-0.4, mean220, std220))

# finding quantiles of normal distribution - norm.ppf (Percent Point Function)
# finding 95% Value at Risk (VaR)
print(norm.ppf(0.05, mu, std))

# "Distribution of a daily and monthly stock return, are rather symmetric about their means, but the tails are fatter.
# Which means there are more outliers that would be expected with normal distributions." Fama and French

plt.xlim(plt_lim[0], plt_lim[1])
plt.ylim(plt_lim[2], plt_lim[3])
plt.show()
