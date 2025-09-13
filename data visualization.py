import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_values_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

def plot_age_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True, bins=30)
    plt.title("Age Distribution")
    plt.show()  

def plot_survival_count(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Survived', data=df)
    plt.title("Survival Count")
    plt.show()

def plot_fare_distribution(df): 
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Fare'], kde=True, bins=30)
    plt.title("Fare Distribution")
    plt.show()

def plot_age_vs_fare(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
    plt.title("Age vs Fare Scatter Plot")
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_pclass_survival(df):

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title("Survival Rate by Passenger Class")
    plt.show()
