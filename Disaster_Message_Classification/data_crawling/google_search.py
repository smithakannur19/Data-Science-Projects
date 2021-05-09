import pandas as pd
import requests
import os
from bs4 import BeautifulSoup
import re
import glob
import os
from fuzzywuzzy import fuzz

import time

#To  get date and time
timestr = time.strftime("%Y%m%d-%H%M%S")



def google_search(query):

	query = query.replace(' ', '+')
	URL = f"https://google.com/search?q={query}"

	# desktop user-agent
	USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
	# mobile user-agent
	MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"

	headers = {"user-agent": USER_AGENT}
	resp = requests.get(URL, headers=headers)
	print (resp.status_code)
	if resp.status_code == 200:
		soup = BeautifulSoup(resp.content, "html.parser")
		results = []
		print (soup)
		for g in soup.find_all('div', class_='g'):
			anchors = g.find_all('a')
			if anchors:
				link = anchors[0]['href']
				title = g.find('h3').text
				item = {
					"title": title,
					"link": link
					}
				results.append(item)
				print (item)        
		return results

list_of_files = glob.glob('./data_crawling/google_input*.csv')
latest_file = max(list_of_files, key=os.path.getctime)

dataframe=pd.read_csv(latest_file)
data_list_dict = dataframe.to_dict()

def match_score(input_c,input_s):
    Ratio = fuzz.ratio(input_c.lower(), input_s.lower())
    Partial_Ratio = fuzz.partial_ratio(input_c.lower(), input_s.lower())
    Token_Sort_Ratio = fuzz.token_sort_ratio(input_c, input_s)
    Token_Set_Ratio = fuzz.token_set_ratio(input_c, input_s)
    print (input_c,input_s)
    print(Ratio)
    print(Partial_Ratio)
    print(Token_Sort_Ratio)
    print(Token_Set_Ratio)
    return (Ratio,Partial_Ratio,Token_Sort_Ratio,Token_Set_Ratio)


def search_result_validate(input_results,onecountry,disasterevent,fromdate):
    search_query=onecountry+" "+disasterevent+" "+"twitter"
    print ("++++++++++++++++++++++")
    print (search_query)
    search_result_list = []
    for values in input_results:
        search_result_dic = {}
        tw_link_check=re.search(r'twitter',values["link"],re.I|re.M)
        if tw_link_check:
            print (values["title"])
            hash_tag_list=re.findall(r"#(\w+)", values["title"])
            hash_tag=''.join(hash_tag_list)
            print (hash_tag)
            print (values["link"])
            (search_result_dic["match_ratio"],search_result_dic["partial_match_ratio"],search_result_dic["token_sort_match_ratio"],search_result_dic["token_set_match_ratio"])=match_score(values["title"],search_query)
            search_result_dic["search_keyword"]=search_query
            search_result_dic["result_title"]=values["title"]
            search_result_dic["result_link"]=values["link"]
            search_result_dic["input_country"]=onecountry
            search_result_dic["input_disaster_event"]=disasterevent
            search_result_dic["input_from_date"]=fromdate
            search_result_dic["hash_tag"]=hash_tag
            search_result_list.append(search_result_dic)
    return(search_result_list)

result_list=[]
for index, row in dataframe.iterrows():
    #print (row)
    disaster_event=row['eventtype']
    country=row['country']
    from_date=row['fromdate']
    print (from_date)
    if disaster_event != 'drought':
        if ',' in country:
            print(country)
            country_list = list(country.split(","))
            for one_country in country_list:
                search_result = google_search(one_country+" "+disaster_event+" "+"twitter")
                result_list.append(search_result_validate(search_result,one_country,disaster_event,from_date))
        else:
            search_result = google_search(country+" "+disaster_event+" "+"twitter")
            result_list.append(search_result_validate(search_result, country,disaster_event,from_date))

df =pd.DataFrame(result_list).stack().apply(pd.Series)
google_output_file_name="./data_crawling/google_output_"+timestr+".csv"
df.to_csv(google_output_file_name,index = None, header=True)

