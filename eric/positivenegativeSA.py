#Import libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from dotenv import load_dotenv
import os


from tweepy import API, OAuth1UserHandler
from tweepy import Cursor
from tweepy.streaming import Stream
from tweepy import OAuthHandler
from tweepy import Stream

import matplotlib.pyplot as plt



load_dotenv()

consumerKey = os.getenv("consumerKey")
consumerSecret = os.getenv("consumerSecret")
accessToken = os.getenv("accessToken")
accessTokenSecret = os.getenv("accessTokenSecret")

# Remove after use
print("1: " + consumerKey)
print("2: " + consumerSecret)
print("3: " + accessToken)
print("4: " + accessTokenSecret)

# Create authentication object
authenticate = tweepy.OAuth1UserHandler(consumerKey, consumerSecret)

# Set access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Create the API object while passing in auth info
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# ///////////////

# Extract 100 tweets from user
# posts = api.user_timeline(screen_name = "BillGates", count= 100, lang = "en", tweet_mode = "extended")

# Print last 5 tweets from user
print("Show the 5 recent tweets: \n")

# Fillers for use
# ///////////////////////////////////////////////////////////////////////////
# posts = ["In February 2020, we had a degree of optimism that the world was better prepared to respond to COVID-19. Two years later, we gathered to discuss lessons learned that might help us prevent the next pandemic.",
#          "A world without tuberculosis is attainable if we deliver effective diagnostics & treatment to all who need it. Fully funding The Global Fund will get us on track to end TB and save 20 million more lives.",
#          "Madeleine Albright made history in so many ways during her lifetime, including as the first female Secretary of State. Today the world lost an advocate for peace and a visionary who believed in our common humanity.",
#          "Every day I’m reminded of how my dad’s wisdom, generosity, and compassion lives on in the many people he influenced and inspired around the world: https://gatesnot.es/36IkLw3",
#          "Sudha Varghese runs a school in Bihar, India that teaches students how to stand up for themselves and see their own potential for greatness. https://gatesnot.es/3wtYe0V"]
#
# i = 1
# for tweet in posts[0:4]:
#     # print(str(i) + ') ' + tweet.full_text + '\n')
#     print(str(i) + ') ' + posts[i] + '\n')
#     i = i + 1
# ////////////////////////////////////////////////////////////////////////////


df = pd.read_csv('train.csv')
print(df)




# Create dataframe with a column for Tweets
# df = pd.DataFrame( [tweet.full_text for tweet in posts], columns=['Tweets']) //Use when api works
# df = pd.DataFrame()
# df['Tweets'] = posts
# print(df.head())


# Clean Text
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#','', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+','', text)

    return text

print('\n')
print("Tweets Cleaned to Remove Extra Text. ie. retweets, links, etc")
print('/////////////////////////////////////////////////////////////')
df['Tweet'] = df['Tweet'].apply(cleanText)
print(df)




# Create function for subjectivity and polarity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Apply functions on df
df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)
df['Polarity'] = df['Tweet'].apply(getPolarity)

# Run subjectivity/ polarity on df
print('\n')
print("Subjectivity and Polarity of df")
print('////////////////////////////////')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df.sample(n=3))

# Plot Word Cloud
allWords = ' '.join([twts for twts in df['Tweet']])
wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Create function for negative, neutral, and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)

plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    if(df['Type'][i] == "Spam"):
        plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color = 'Red')
    else:
        plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')

plt.title("Sentiment Analysis")
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


#Get Positive Percentage
ptweets = df[df.Analysis == "Positive"]
ptweets = ptweets['Tweet']
print(round((ptweets.shape[0]/ df.shape[0] * 100), 1))

#Get Negative Percentage
ptweets = df[df.Analysis == "Negative"]
ptweets = ptweets['Tweet']
print(round((ptweets.shape[0]/ df.shape[0] * 100), 1))

df['Analysis'].value_counts()

plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()