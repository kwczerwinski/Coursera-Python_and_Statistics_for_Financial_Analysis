# rolling 2 6-sided dices
import pandas as pd
import matplotlib.pyplot as plt

die = pd.DataFrame([1, 2, 3, 4, 5, 6])

dice_throws = die.sample(2, replace=True)
print(dice_throws)
print(dice_throws.sum())
print('Sum of dice:', dice_throws.sum().loc[0])

trial = 10000
outcomes = [die.sample(2, replace=True).sum().loc[0] for _ in range(trial)]
# print(outcomes)

unique_values = [int(x) for x in set(outcomes)]
print(unique_values)

distribution = {}
for v in unique_values:
    distribution[v] = outcomes.count(v)

print(distribution)

plt.plot(list(distribution.keys()), list(distribution.values()))
plt.show()
