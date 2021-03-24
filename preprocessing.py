import pandas as pd

df = pd.read_csv('datasets/mnist_train.csv')
df_norm = (df.iloc[2:, :] - df.iloc[2:, :].mean())/df.iloc[2:, :].std()
df_norm.to_csv('train_standarized.csv', index=False, header=False)
