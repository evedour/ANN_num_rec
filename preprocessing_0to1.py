import pandas as pd

df = pd.read_csv('datasets/mnist_train.csv')
df_norm = (df.iloc[2:, :])/255
df_norm.to_csv('train_normalized.csv', index=False, header=False)
