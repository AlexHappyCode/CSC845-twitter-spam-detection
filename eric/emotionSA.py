# import libraries
# https://pypi.org/project/NRCLex/
from nrclex import NRCLex
import pandas as pd
import re
# Assigning Test Words
text = ['Boston bombing survivor s charity gives 1st artificial limb #DemnDebate #DemDebate #tcot', '2 Attacked by Hawks on NYC College Campus: Officials https://t.co/tVL2xHitu0',
        'North Side Chase Ends With Crash Into Gas Station Under Construction  #news', '"AND TISOY @aldenrichards02  POSTED LETTER LIKE M #ALDUBLoversInITALYpic.twitter.com/rUYNKR3vw5"']

for i in range(len(text)):
    # create emotion objects
    emotion = NRCLex(text[i]) # fear, anger, anticip, trust, surprise, positive, negative, sadness, disgust, joy
    # print(emotion.affect_frequencies['fear'])
    # print(type(emotion.affect_frequencies))
    print(emotion.sentences)
    print(emotion.affect_frequencies)

    # Classify emotion
    # print('\n\n', text[i], ': ', emotion.affect_frequencies.items()[0])
    # print(type(emotion.affect_frequencies))
    # print(next(iter(emotion.affect_frequencies.items())))

def nrclexFunctionFear(tweet):
    emotion = NRCLex(tweet)
    return emotion.affect_frequencies['fear']

def nrclexFunctionAnger(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['anger']

def nrclexFunctionAnticip(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['anticip']

def nrclexFunctionTrust(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['trust']

def nrclexFunctionSurprise(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['surprise']

def nrclexFunctionPositive(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['positive']

def nrclexFunctionNegative(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['negative']

def nrclexFunctionSadness(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['sadness']

def nrclexFunctionDisgust(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['disgust']

def nrclexFunctionJoy(tweet):
    emotion = NRCLex(tweet)
    return  emotion.affect_frequencies['joy']


df = pd.read_csv('train.csv')


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





# applied_df = df.apply(lambda x : NRCLex(df['Tweet'].affect_frequencies), axis = 'columns', result_type = 'expand')
# df['fear','anger','anticip'] = df['Tweet'].apply(nrclexFunction)
# df = pd.concat([df, applied_df], axis = 'columns')

# TODO: REPEAT THIS WITH OTHER VALUES
# TODO: FIGURE CORRECT WAY LATER

# # Generate Values for Sentiment with NRCLex
# df['fear'] = df['Tweet'].apply(nrclexFunctionFear)
# df['anger'] = df['Tweet'].apply(nrclexFunctionAnger)
# df['anticip'] = df['Tweet'].apply(nrclexFunctionAnticip)
# df['trust'] = df['Tweet'].apply(nrclexFunctionTrust)
# df['surprise'] = df['Tweet'].apply(nrclexFunctionSurprise)
# df['positive'] = df['Tweet'].apply(nrclexFunctionPositive)
# df['negative'] = df['Tweet'].apply(nrclexFunctionNegative)
# df['sadness'] = df['Tweet'].apply(nrclexFunctionSadness)
# df['disgust'] = df['Tweet'].apply(nrclexFunctionDisgust)
# df['joy'] = df['Tweet'].apply(nrclexFunctionJoy)

pd.set_option('display.max_columns', None)

# print()


# print(df)
print(df.head())


# https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns


# TODO: https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
