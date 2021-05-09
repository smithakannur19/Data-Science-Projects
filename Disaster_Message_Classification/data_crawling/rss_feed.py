from urllib.request import urlopen
import xml.etree.ElementTree as ET
import pandas as pd
import time

#To  get date and time
timestr = time.strftime("%Y%m%d-%H%M%S")

#Reading the rss feed
url = "https://www.gdacs.org/xml/rss_7d.xml"
url_to_open = urlopen(url).read()

#Svaing the feed data as xml
filename = "./data_crawling/rss_24h_"+timestr+".xml"
file_ = open(filename, 'wb')
file_.write(url_to_open)
file_.close()

#Processing the xml data
tree = ET.parse(filename)
root = tree.getroot()

#Dictnory of disaster event and there respective code
code_dict={"DR":"drought","TC":"cyclone","FL":"flood",
"EQ":"earthquake","VO":"Volcanic eruption"}
disaster_list=[]

#Looping through the xml data
for item in root.iter('item'):
    disaster_dict={}
    for subelement in item:
        #Extracting the required data
        if subelement.tag == '{http://www.gdacs.org}fromdate':
            disaster_dict["fromdate"] = subelement.text
            # print (subelement.text)
        if subelement.tag == '{http://www.gdacs.org}country':
            disaster_dict["country"]=subelement.text
            #print (subelement.text)
        if subelement.tag == '{http://www.gdacs.org}eventtype':
            disaster_dict["eventtype"] = code_dict[subelement.text]
            #print (subelement.text)
        if subelement.tag == '{http://www.gdacs.org}alertlevel':
            disaster_dict["alertlevel"] = subelement.text
            #print (subelement.text)
    disaster_list.append(disaster_dict)
    print (disaster_list)
# converting the list to dataframe
df = pd.DataFrame(disaster_list)

#Creating a datframe containing only orange & red disaster levels
twitter_input=df.loc[df['alertlevel'] != 'Green']

#csv filenames for storing the data
google_input_file_name="./data_crawling/google_input_"+timestr+".csv"
all_incident_file_name="./data_crawling/alert_"+timestr+".csv"

# writing as csv
twitter_input.to_csv(google_input_file_name,index = None, header=True)
df.to_csv(all_incident_file_name,index = None, header=True)

