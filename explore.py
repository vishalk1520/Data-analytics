import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("\n--- BASIC INFO ---")
    print(df.info())
    print("\n--- SUMMARY STATS ---")
    print(df.describe(include="all"))

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- SURVIVAL RATE ---")
    print(df['Survived'].value_counts(normalize=True))

    sns.countplot(x='Survived', data=df)
    plt.title("Survival Counts")
    plt.show()

    sns.histplot(df['Age'].dropna(), bins=30, kde=True)
    plt.title("Age Distribution")
    plt.show()

    sns.histplot(df['Fare'], bins=40, kde=True)
    plt.title("Fare Distribution")
    plt.show()

    print("\n--- Survival by Sex ---")
    print(df.groupby('Sex')['Survived'].mean())

    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title("Survival Rate by Sex")
    plt.show()

    print("\n--- Survival by Pclass ---")
    print(df.groupby('Pclass')['Survived'].mean())

    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title("Survival Rate by Passenger Class")
    plt.show()

    sns.boxplot(x='Survived', y='Age', data=df)
    plt.title("Age vs Survival")
    plt.show()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
