import pandas as pd

df = pd.read_csv('datasets/mnist_train.csv')
df_norm = (df.iloc[2:, :])/255
rslt =  pd.concat([df_norm, df.iloc[:, 1]], axis=1)
rslt.to_csv('train_normalized.csv', index=False, header=False)