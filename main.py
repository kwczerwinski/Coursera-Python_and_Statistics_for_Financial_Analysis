# rolling 2 6-sided dices
import pandas as pd

die = pd.DataFrame([1, 2, 3, 4, 5, 6])

dice_throws = die.sample(2, replace=True)
print(dice_throws)
print(dice_throws.sum())
print('Sum of dice:', dice_throws.sum().loc[0])
