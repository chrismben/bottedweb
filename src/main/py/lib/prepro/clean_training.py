#source: https://github.com/devspotlight/Reddit-Dashboard-ML/blob/master/clean_data_bots_trolls.ipynb

import pandas as pd
import numpy as np
import psycopg2
import json
import datetime as dt
import difflib
from textblob import TextBlob


with open('lib/data/reddit.csv', encoding="utf8") as f:
    my_data = pd.read_csv(f, sep=',', dtype={
        "banned_by": str,
        "no_follow": bool,
        "link_id": str,
        "gilded": bool,
        "author": str,
        "author_verified": bool,
        "author_comment_karma": np.float64,
        "author_link_karma": np.float64,
        "num_comments": np.float64,
        "created_utc": np.float64,
        "score": np.float64,
        "over_18": bool,
        "body": str,
        "downs": np.float64,
        "is_submitter": bool,
        "num_reports": np.float64,
        "controversiality": np.float64,
        "quarantine": str,
        "ups": np.float64,
        "is_bot": bool,
        "is_troll": bool,
        "recent_comments": str})
    
# print out an example JSON object for testing the API
row = my_data[my_data.author=='AutoModerator'].iloc[[10]]
row.recent_comments = pd.read_json(row.recent_comments.values[0], dtype={
        "banned_by": str,
        "no_follow": bool,
        "link_id": str,
        "gilded": np.float64,
        "author": str,
        "author_verified": bool,
        "author_comment_karma": np.float64,
        "author_link_karma": np.float64,
        "num_comments": np.float64,
        "created_utc": np.float64,
        "score": np.float64,
        "over_18": bool,
        "body": str,
        "downs": np.float64,
        "is_submitter": bool,
        "num_reports": np.float64,
        "controversiality": np.float64,
        "quarantine": bool,
        "ups": np.float64}).to_json(orient='records')
row.to_json(orient='records').replace('\'','').replace('\\\\','\\').replace('[{','{').replace('}]','}')   


# delete columns that have missing data or won't have meaningful values in real-time data
columns = ['banned_by', 'downs', 'quarantine', 'num_reports', 'num_comments', 'score', 'ups', 'controversiality', 'gilded']
my_data.drop(columns, inplace=True, axis=1)

# drop duplicates
my_data.drop_duplicates(inplace=True)

# format columns
my_data['created_utc'] = pd.to_datetime(my_data['created_utc'].values, unit='s')
my_data['body'] = my_data['body'].str.slice(stop=200).fillna('')

# add our new stats columns
my_data['recent_num_comments'] = pd.Series(np.zeros(len(my_data.index), np.int64))
my_data['recent_num_last_30_days'] = pd.Series(np.zeros(len(my_data.index), np.int64))
my_data['recent_avg_no_follow'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_gilded'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_responses'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_percent_neg_score'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_score'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_min_score'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_controversiality'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_ups'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_diff_ratio'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_max_diff_ratio'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_avg_sentiment_polarity'] = pd.Series(np.zeros(len(my_data.index), np.float64))
my_data['recent_min_sentiment_polarity'] = pd.Series(np.zeros(len(my_data.index), np.float64))

# Count num of bots and trolls
bots = my_data['is_bot']
trolls = my_data['is_troll']
normies = my_data[(my_data.is_bot == False) & (my_data.is_troll == False)]
print("Number of bot comments: ", bots.sum())
print("Number of troll comments:", trolls.sum())
print("Number of normal comments:", len(normies))

bot_authors = my_data[my_data['is_bot'] == True][['author']]
troll_authors = my_data[my_data['is_troll'] == True][['author']]
print("Number of bot authors: ", len(np.unique(bot_authors)))
print("Number of troll authors:", len(np.unique(troll_authors)))

# Num of users
users = my_data['author'].values
num_of_users = np.unique(users)
print("Number of total authors: ", len(num_of_users))

# Set fractions between the user classes
print("\nFixing ratios between classes")
data = my_data[my_data['is_troll']]
my_data = data.append(my_data[my_data['is_bot']].sample(n=len(data)*2))

def diff_ratio(_a, _b):
    return difflib.SequenceMatcher(a=_a,b=_b).ratio()

def last_30(a, b):
    return a - dt.timedelta(days=30) < pd.to_datetime(b, unit='s')

num = 0;

def calc_stats(comment):
    # track progress
    global num 
    num += 1
    if(num % 1000 == 0): print(num)
        
    recent_comments = pd.read_json(comment['recent_comments'], dtype={
        "banned_by": str,
        "no_follow": bool,
        "link_id": str,
        "gilded": np.float64,
        "author": str,
        "author_verified": bool,
        "author_comment_karma": np.float64,
        "author_link_karma": np.float64,
        "num_comments": np.float64,
        "created_utc": np.float64,
        "score": np.float64,
        "over_18": bool,
        "body": str,
        "downs": np.float64,
        "is_submitter": bool,
        "num_reports": np.float64,
        "controversiality": np.float64,
        "quarantine": bool,
        "ups": np.float64})
    comment['recent_num_comments'] = len(recent_comments)
    
    if(len(recent_comments) > 0):
        comment['recent_num_last_30_days'] = recent_comments['created_utc'].apply(lambda x: last_30(comment['created_utc'], x)).sum()
        comment['recent_avg_no_follow'] = recent_comments['no_follow'].mean()
        comment['recent_avg_gilded'] = recent_comments['gilded'].mean()
        comment['recent_avg_responses'] = recent_comments['num_comments'].mean()
        comment['recent_percent_neg_score'] = recent_comments['score'].apply(lambda x: x < 0).mean() * 100
        comment['recent_avg_score'] = recent_comments['score'].mean()
        comment['recent_min_score'] = recent_comments['score'].min()
        comment['recent_avg_controversiality'] = recent_comments['controversiality'].mean()
        comment['recent_avg_ups'] = recent_comments['ups'].mean()
        diff = recent_comments['body'].str.slice(stop=200).fillna('').apply(lambda x: diff_ratio(comment['body'], x))
        comment['recent_avg_diff_ratio'] = diff.mean()
        comment['recent_max_diff_ratio'] = diff.max()
        scores = recent_comments['body'].append(pd.Series(comment['body'])).apply(lambda x: TextBlob(x).sentiment.polarity)
        comment['recent_avg_sentiment_polarity'] = scores.mean()
        comment['recent_min_sentiment_polarity'] = scores.min()
        
    return comment

new_data = my_data.apply(calc_stats, axis=1)

# delete NA values
new_data=new_data[new_data.recent_min_sentiment_polarity.isna() == False]

def setTarget(x):
    if(x.is_bot): 
        return 'bot'
    elif(x.is_troll): 
        return 'troll' 
    else:
        return 'normal'

# Create one column with the target training label
new_data['target'] = new_data.apply(lambda x: setTarget(x), axis=1)

# Delete is_bot and is_troll collumns and add targets column
columns = ['is_bot', 'is_troll']
new_data.drop(columns, inplace=True, axis=1)

# Delete recent_comments to save space
columns = ['recent_comments']
new_data.drop(columns, inplace=True, axis=1)

new_data.to_csv('lib/data/my_clean_data_training.csv', sep=',', index=False)
print("The data cleaning finished correctly!!!")


