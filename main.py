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

# Probability Density Function (PDF)

plt.show()
