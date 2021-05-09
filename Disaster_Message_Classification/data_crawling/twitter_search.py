from TwitterSearch import *
import pandas as pd
import re
from fuzzywuzzy import fuzz
import time
import json
from textblob import TextBlob
import glob
import os
import spacy
nlp = spacy.load("en_core_web_sm")

def match_score(input_c,input_s):
    Ratio = fuzz.ratio(input_c.lower(), input_s.lower())
    Partial_Ratio = fuzz.partial_ratio(input_c.lower(), input_s.lower())
    Token_Sort_Ratio = fuzz.token_sort_ratio(input_c, input_s)
    Token_Set_Ratio = fuzz.token_set_ratio(input_c, input_s)
    #print(Ratio)
    #print(Partial_Ratio)
    #print(Token_Sort_Ratio)
    #print(Token_Set_Ratio)
    return (Ratio,Partial_Ratio,Token_Sort_Ratio,Token_Set_Ratio)

def tweet_collect(hashstring,in_country,disaster):
    tweets_list=[]
    #time.sleep(2)
    try:
        search_string="#"+hashstring
        tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
        tso.set_language('en')
        tso.add_keyword(search_string)
        ts = TwitterSearch(
            consumer_key='LheTHAR7DSfhkIqYiBdlA',
            consumer_secret='F8uj4jsQv7THfjs1fUf7iLDTlgQUcneJIEhEXgO6A',
            access_token='1282607706-yAOJ6ZQ8zLJrTPK1LxzEJ4yfgU24EwHDK64LFAu',
            access_token_secret='UfpSAvmUyio0ydV2mn5kBz3fP7A6c5JHmlNyGizFVvYtG',
            verify=False
        )
        #print("Current rate-limiting status: %s" % ts.get_metadata()['x-rate-limit-remaining'])
        for tweet in ts.search_tweets_iterable(tso):
            #print(tweet)
            with open(twitter_raw_filename, 'a', encoding='utf-8') as f:
                json.dump(tweet, f, ensure_ascii=False)
            f.close()
            tweet_dict={}
            tweet_dict["tweet_user_name"] = tweet['user']['name']
            tweet_dict["tweet_user_screen"] = tweet['user']['screen_name']
            tweet_dict["follower_count"] = tweet['user']['followers_count']
            tweet_dict["user_created"] = tweet['user']['created_at']
            tweet_dict["tweet_count"] = tweet['user']['statuses_count']
            tweet_dict["following_status"] = tweet['user']['following']
            tweet_dict["friends_count"] = tweet['user']['friends_count']
            tweet_dict["favourites_count"] = tweet['user']['favourites_count']
            tweet_dict["description"] = tweet['user']['description']
            tweet_dict["user_verified"] = tweet['user']['verified']
            tweet_dict["user_time_zone"] = tweet['user']['time_zone']
            tweet_dict["user_location"] = tweet['user']['location']
            tweet_dict["user_tweet"]=tweet['text']
            hash_tag_list = re.findall(r"#(\w+)", tweet['text'])
            hash_tag = ','.join(hash_tag_list)
            tweet_dict["user_created"]=tweet['created_at']
            tweet_dict["hash_tag"]=hash_tag
            clean_the_tag = re.compile('<.*?>')
            cleantext = re.sub(clean_the_tag, '', tweet['source'])
            tweet_dict["tweet_source"] = cleantext
            doc = nlp(tweet['text'])
            location_from_tweets_list=[]
            for token in doc.ents:
                if token.label_ == "GPE" or token.label_ == "LOC" and token.label_ != "":
                    print (token.text,token.label_)
                    location_from_tweets_list.append(token.text)
            location_from_tweets=",".join(location_from_tweets_list)
            tweet_dict["location_from_tweets"] = location_from_tweets
            tweet_sentiment=TextBlob(tweet['text'])
            tweet_dict["tweet_sentiment_polarity"]=tweet_sentiment.polarity
            tweet_dict["tweet_sentiment_subjectivity"]=tweet_sentiment.subjectivity
            (tweet_dict["match_ratio"],tweet_dict["country_partial_match_ratio"],tweet_dict["token_sort_match_ratio"],tweet_dict["token_set_match_ratio"],)=match_score(tweet['text'],in_country+" "+disaster)
            (tweet_dict["location_from_tweets_match_ratio"],tweet_dict["location_from_tweets_country_partial_match_ratio"],tweet_dict["location_from_tweets_token_sort_match_ratio"],tweet_dict["location_from_tweets_token_set_match_ratio"],)=match_score(location_from_tweets,in_country)
            tweet_dict["input_disaster_location"]=in_country
            tweet_dict["input_disaster"]=disaster
            print(tweet_dict)
            tweets_list.append(tweet_dict)
        return tweets_list
    except TwitterSearchException as e:
        print(e)



#To  get date and time
timestr = time.strftime("%Y%m%d-%H%M%S")
twitter_raw_filename="./data_crawling/twitter_raw_data"+timestr+".json"
result_list=[]
list_of_files = glob.glob('./data_crawling/google_output*.csv')
latest_file = max(list_of_files, key=os.path.getctime)
dataframe=pd.read_csv(latest_file)
for index, row in dataframe.iterrows():
    #print(row)
    if type(row['hash_tag']) == str:
        result_list.append(tweet_collect(row['hash_tag'],row['input_country'],row['input_disaster_event']))
    #input()

df =pd.DataFrame(result_list).stack().apply(pd.Series)
twitter_output_file_name="./data_crawling/twitter_output_"+timestr+".csv"
df.to_csv(twitter_output_file_name,index = None, header=True)
