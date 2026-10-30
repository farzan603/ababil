import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')

# data clean

df_cleaned = df.copy()



df_cleaned.drop_duplicates(inplace=True)

df_cleaned['sex'] = df['sex'].map({'male':0, 'female':1})

df_cleaned['smoker'] = df['smoker'].map({'yes':0, 'no':1})

df_cleaned.rename(columns={'sex':'is_female',
                           'smoker':'is_smoker',
                           }, inplace=True)

df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)

df_cleaned = df_cleaned.astype(int)
print(df_cleaned.shape)

# feature engineering

bins = [0,18.5,25,30, np.inf]

labels = ["underweight","normal","overweight","obese"]

df_cleaned['bmi_categoery'] = pd.cut(df_cleaned['bmi'], bins=bins,labels=labels,right=False)

df_cleaned = pd.get_dummies(df_cleaned,columns=['bmi_categoery'],drop_first=True)
df_cleaned = df_cleaned.astype(int)

cols = ['age','bmi','children']
scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

target = "charges"

for column in df_cleaned.columns:
    if column != target:
        correlation = df_cleaned[column].corr(
            df_cleaned[target]
        )

        print(f"{column:<30} : {correlation:.4f}")

