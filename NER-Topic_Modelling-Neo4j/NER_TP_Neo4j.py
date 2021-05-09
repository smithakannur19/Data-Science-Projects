#!/usr/bin/env python
# coding: utf-8

# **Problem:
# Generating and storing structured data from unstructured text.**
#  
# **Task:
# Take text data, perform NER and Topic extraction, then store these entities and topics in a Neo4j database. Entities and topics are considered related if they appear in the same document. We would like to use this graph to determine the relations between topics and entities. You should write a Neo4j query that can return the 100 pairs of entities which are most closely related, and the 100 pairs of topics that are most closely related. You can decide the precise measure of relatedness you use.**
#  
# **Data:
# A corpus of one million news articles can be found at: https://research.signal-ai.com/newsir16/signal-dataset.html**

# **Approached Solution:**
# 
# **I have taken the data from the link mentioned above. I have considered top 3000 documents(content column is used) for compeleting the task.** 
# **Path to the data and neo4j credentials can be given in configuration file.** 
# 1. NER using spacy is performed on 3k documents.
# 2. "Content" of the column is preprocessed.
# 3. Topic modelling is done using LDA
# 4. 500 topics has been considered and top 300 keywords from each topic is taken. 
#    (Topic name is not manually assigned. Its in Topic Number - Topic 0, Topic 1 etc)
# 5. Entities within the keywords is taken (intersection of NER results and Topic Keywords for each document)
# 6. Using neo4j, Entity nodes and Topic nodes are represented - using py2neo v3.
# 7. Top 100 Topic pairs and 100 Top entity pairs are extracted by doing jaccard similarity. 
# 8. The Topic pairs(sheet1) and entity pairs(sheet2) are written to the excel sheet - Output.xlsx

# In[32]:


#import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from tqdm import tqdm
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import en_core_web_lg
import ast
from py2neo import Graph, Node, Relationship
from configparser import ConfigParser 
from itertools import combinations


# In[24]:


# read the config file
config = ConfigParser() 
config.read('config.ini')


# In[ ]:


# for NER 
nlp = en_core_web_lg.load()


# In[2]:


# Loading data
data_path=config["data"]["data_path"]
df = joblib.load(data_path)
df = df.head(3000)
df.head()


# In[3]:


# apply NER
#spacy - named entities on content column
get_entities = lambda row:[[ent.label_,ent.text] for ent in nlp(row).ents]
df["content_entities"]=df["content"].apply(get_entities)


# In[4]:


df = df[["content","content_entities"]]
df.head(3)


# In[5]:


# Data Preprocessing

# remove stopwords
cachedStopWords = stopwords.words("english")
df["processed_content"] = df["content"].str.lower()
df["processed_content"] = df["processed_content"].apply(lambda x: ' '.join([word for word in x.split() if word not in (cachedStopWords)]))

# lemmatization
sent_lemma=[]
corpus = df["processed_content"]
for text in corpus:
    words = word_tokenize(text)
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(token,pos='v') for token in words]
    sent_lemma.append(' '.join(text))

df["processed_content"] = sent_lemma

#removing words having length upto 2
df["processed_content"] = df["processed_content"].str.split().map(lambda sl: " ".join(s for s in sl if len(s) > 2))


# In[6]:


df.head(3)


# In[7]:


# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1,3),stop_words='english', lowercase=True)
data_vectorized = vectorizer.fit_transform(df["processed_content"])


# In[8]:


#vectorizer.get_feature_names()


# In[9]:


NUM_TOPICS = 500

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)


# In[10]:


def get_topic_num(row):
    """
    This function predicts which topic does the document belongs to.
    Parameters: document
    return: Topic number
    """
    result=(lda.transform(vectorizer.transform([row]))[0]).tolist()
    pos=result.index(max(result))
    return pos


# In[11]:


df["topic_number"]=df["processed_content"].apply(lambda row:get_topic_num(row))


# In[12]:


df.head()


# In[13]:


model=lda
vectorizer=vectorizer
top_n=300


# In[14]:


def get_topic_keywords(num):
    """
    This function gets the top 300 keywords of the respective topic to which the document belongs to.
    Parameters: topic number
    return: keywords of the coresponding topic
    """
    kw=list(enumerate(model.components_))[num][1]
    topic_kw=[]
    for i in kw.argsort()[:-top_n - 1:-1]:
        topic_kw.append(vectorizer.get_feature_names()[i])
    return topic_kw


# In[15]:


tqdm.pandas()
df["topic_keywords"]=df["topic_number"].progress_apply(lambda num:get_topic_keywords(num))


# In[16]:


df.head(3)


# In[37]:


# neo4j configuration/credentials 
uri=config["neo4j"]["uri"]
user=config["neo4j"]["user"]
password=config["neo4j"]["password"]
graph = Graph(uri=uri, user=user, password=password)


# In[39]:


gb = graph.begin()

topic_entity_name=[]
for index, row in tqdm(df.iterrows()):
    # get the topic number of the doc and create node
    topic=row["topic_number"]
    topic_node = graph.nodes.match("TopicNumber", name="Topic_"+str(topic)).first()
    if topic_node is None:
        topic_node = Node("TopicNumber", name="Topic_"+str(topic))
        graph.create(topic_node)
    
    topic_keywords=row["topic_keywords"]
    
    temp=[]
    #creating nodes and relationship between entities present in topic and topic number to which a corresponding doc belongs to
    content_entities=row["content_entities"]
    for elem in content_entities:
        if elem[1].lower() in topic_keywords:
            if elem not in temp:
                temp.append(elem)
            entity_node = graph.nodes.match(elem[0], name=elem[1]).first()
            if entity_node is None:
                entity_node = Node(elem[0], name=elem[1])
                graph.create(entity_node)
            relation1 = Relationship(topic_node, "HAS_ENTITIES", entity_node)
            relation2 = Relationship(entity_node, "BELONGS_TO", topic_node)
            graph.create(relation1)
            graph.create(relation2)
    
    topic_entity_name.append(temp)

gb.commit()


# In[73]:


df["topic_entities"]=topic_entity_name
df.head(3)


# In[41]:


unique_topic_num = list(set(list(df["topic_number"])))
topic_comb = list(combinations(unique_topic_num, 2))


# In[42]:


gb = graph.begin()

topic_similarity=[]
for i in tqdm(topic_comb):

    result=graph.run("""MATCH (tp1:TopicNumber {name: {t1}})-[:HAS_ENTITIES]->(topic1)
    WITH tp1, collect(id(topic1)) AS topic_tp1
    MATCH (tp2:TopicNumber {name: {t2}})-[:HAS_ENTITIES]->(topic2)
    WITH tp1, topic_tp1, tp2, collect(id(topic2)) AS topic_tp2
    RETURN tp1.name AS from,
           tp2.name AS to,
           algo.similarity.jaccard(topic_tp1, topic_tp2) AS similarity""",t1="Topic_"+str(i[0]),t2="Topic_"+str(i[1])).data()
    #print(list(result))
    if len(list(result))!=0:
        topic_similarity.append(result[0])
    #break

gb.commit()


# In[45]:


topic_sim_df=pd.DataFrame(topic_similarity,columns=["from","to","similarity"])
topic_sim_df=topic_sim_df.sort_values(by='similarity', ascending=False)
topic_100_pairs=topic_sim_df.head(100)
topic_100_pairs.head(3)


# In[48]:


topic_num=list(df["topic_number"])
topic_kw=list(df["topic_keywords"])
dictionary = dict(zip(topic_num, topic_kw))


# In[56]:


from_topic=[]
from_topic_kw=[]
to_topic=[]
to_topic_kw=[]
for index, row in topic_100_pairs.iterrows():
    from_num=row["from"].split('_')[1]
    from_kw=dictionary[eval(from_num)]
    to_num=row["to"].split('_')[1]
    to_kw=dictionary[eval(to_num)]
    from_topic.append(from_num)
    from_topic_kw.append(from_kw)
    to_topic.append(to_num)
    to_topic_kw.append(to_kw)


# In[65]:


topic_kw_100_pairs=pd.DataFrame()
topic_kw_100_pairs["From_Topic_Number"]=from_topic
topic_kw_100_pairs["From_Topic_Keywords"]=from_topic_kw
topic_kw_100_pairs["To_Topic_Number"]=to_topic
topic_kw_100_pairs["To_Topic_Keywords"]=to_topic_kw
topic_kw_100_pairs["Similarity"]=list(topic_100_pairs["similarity"])


# In[76]:


topic_kw_100_pairs.head(3)


# In[74]:


unique_topic_ent=list(df["topic_entities"])
flat_list_ent=[]
for i in unique_topic_ent:
    if len(i)!=0:
        for j in i:
            flat_list_ent.append(j[1])

unique_topic_ent=list(set(flat_list_ent))
topic_ent_com = combinations(unique_topic_ent, 2)
topic_ent_com = list(topic_ent_com)


# In[75]:


gb = graph.begin()

entity_similarity=[]
for i in tqdm(topic_ent_com):
    
    node_label1=graph.run("""MATCH (r) WHERE r.name={n} RETURN ID(r), labels(r)""",n=i[0]).data()   
    node_label2=graph.run("""MATCH (r) WHERE r.name={n} RETURN ID(r), labels(r)""",n=i[1]).data()

    for n1 in node_label1:
        for n2 in node_label2:
            result=graph.run("""MATCH (e1:"""+n1['labels(r)'][0]+"""{name: {ent1}})-[:BELONGS_TO]->(entity1)
                                WITH e1, collect(id(entity1)) AS e1_entity
                                MATCH (e2:"""+n2['labels(r)'][0]+"""{name: {ent2}})-[:BELONGS_TO]->(entity2)
                                WITH e1, e1_entity, e2, collect(id(entity2)) AS e2_entity
                                RETURN e1.name AS from,
                                       e2.name AS to,
                                       algo.similarity.jaccard(e1_entity, e2_entity) AS similarity""",ent1=i[0],ent2=i[1]).data()
    
            if len(result)!=0:
                entity_similarity.append(result[0])
                
gb.commit()


# In[77]:


entity_sim_df=pd.DataFrame(entity_similarity,columns=["from","to","similarity"])
entity_sim_df=entity_sim_df.sort_values(by='similarity', ascending=False)
entity_100_pairs=entity_sim_df.head(100)
entity_100_pairs.head(3)


# In[78]:


from pandas import ExcelWriter

writer = ExcelWriter('Output.xlsx')
topic_kw_100_pairs.to_excel(writer,'Topic Pairs')
entity_100_pairs.to_excel(writer,'Entity Pairs')
writer.save()


# In[ ]:




