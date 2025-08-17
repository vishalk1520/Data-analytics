import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

print("Initial Data Info:")
print(df.info())
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  

df.drop_duplicates(inplace=True)

df['Sex'] = df['Sex'].str.lower()
df['Embarked'] = df['Embarked'].str.upper()

df.loc[df['Age'] > 80, 'Age'] = 80

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


print("\nAfter Cleaning:")
print(df.isnull().sum())
print(df.head())

sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap After Cleaning")
plt.show()


sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Age Distribution After Cleaning")
plt.show()

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()
