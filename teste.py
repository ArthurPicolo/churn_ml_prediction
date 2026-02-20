#%%
# Import de "novos" dados
import pandas as pd

df = pd.read_csv("data/abt_churn.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(25)
amostra = amostra.drop('flagChurn', axis=1)

#%%
amostra.to_csv('test.csv', index=False)
# %%
