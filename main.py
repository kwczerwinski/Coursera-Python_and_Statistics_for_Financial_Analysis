import pandas as pd

data = pd.read_csv('data/TRADEGATE%DGD=D.csv', index_col=0, sep='\t')

print(type(data))
print(data)
