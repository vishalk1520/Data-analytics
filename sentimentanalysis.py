import pandas as pd
from textblob import TextBlob

# Load the dataset
df = pd.read_csv(r'C:\Users\Vishal Kanojiya\OneDrive\data analytics\IMDB Dataset.csv')


# Function to determine sentiment
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
df['sentiment'] = df['review'].apply(get_sentiment)

# Save the results
df.to_csv(r'C:\Users\Vishal Kanojiya\OneDrive\data analytics\imdb_with_sentiment.csv', index=False)


print("Sentiment analysis completed and saved.")
