import pandas as pd

df = pd.read_csv('datasets/mnist_train.csv')
df_norm = (df.iloc[2:, :] )/255
rslt = pd.concat([df_norm, df.iloc[1, :]]) #adds label row
rslt.to_csv('example.csv', index=False, header=False)
