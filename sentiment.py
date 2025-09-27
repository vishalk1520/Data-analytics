import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with sentiment results
df = pd.read_csv(r'C:\Users\Vishal Kanojiya\OneDrive\data analytics\imdb_with_sentiment.csv')

# Count the number of each sentiment
sentiment_counts = df['sentiment'].value_counts()

# Plot
plt.figure(figsize=(8,5))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Movie Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(6,6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Movie Reviews')
plt.ylabel('')
plt.show()

