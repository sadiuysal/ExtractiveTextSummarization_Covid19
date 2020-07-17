#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
import necessary libraries
"""
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string
from nltk.tokenize import word_tokenize
import json
import os
import math
import re
import numpy as np
from scipy import spatial
import pandas


# In[2]:


"""
reads topicID's relevant documentIDs
"""
def read_relevance_file():
    relevant_docId_dict={} #stores list of documentID's for given topicId
    f=open("NecessaryFiles/qrels-rnd1.txt", "r")
    if f.mode == 'r':
        lines =f.read().splitlines()
    for line in lines:
        columns=line.split(" ")
        topic_id=int(columns[0])
        doc_id=columns[3]
        rel_judgement=int(columns[-1])
        if (topic_id==1 or topic_id==13 or topic_id==17) and rel_judgement==2:
            if relevant_docId_dict.get(topic_id) is None:
                relevant_docId_dict[topic_id]=[doc_id]
            else:
                relevant_docId_dict[topic_id].append(doc_id)
    return relevant_docId_dict
              


# In[3]:


"""
reads docID's relevant abstracts to given dict
"""
def find_documents_abstracts(docId_list):
    #reads topicID's relevant documentIDs to given dict
    docId_abstract_dict={}
    df = pandas.read_csv('NecessaryFiles/04-10-mag-mapping.csv', index_col='cord_uid')
    for docId in docId_list:
        docId_abstract_dict[docId]=df.loc[docId]["abstract"]
    return docId_abstract_dict
"""
finds document's abstract's sentences
"""
def find_documents_sentences(docId_list):
    #reads docsID's sentences and map
    sentences=[]
    for docId in docId_list:
        sentences+=sent_tokenize(docId_abstract_dict[docId])
    return sentences


        


# In[4]:



"""
Exclude non-alphanumeric characters from beginning and end of the string
"""
def strip_nonalnum_re(word):
    return re.sub(r"^\W+|\W+$", "", word)
"""
Fill references in the source file
"""
def fill_references(text,refs):
    for ref in refs:
        #print(ref)
        if ref["ref_id"] is None:
            continue
        if not ref.get("mention") is None:
            text=text.replace(ref["mention"], ref["ref_id"])
        elif not ref.get("text") is None:
            text=text.replace(ref["text"], ref["ref_id"])
    return text


# In[5]:




"""
Finds all files in the given directory and process to get word_idf_dict
"""
def read_process_docs(docs_folder_path_and_selection):
    wordDocfreq={}
    docs_folder_path=docs_folder_path_and_selection[0]
    json_selection=docs_folder_path_and_selection[1]
    tot_doc_count=0
    for root, dirs, files in os.walk(docs_folder_path):
        for file in files:
            if file.endswith(".json"):
                if tot_doc_count%1000==0:
                    print(tot_doc_count)
                tot_doc_count+=1
                file_path=os.path.join(root, file)
                docWordSet=set() # to store word occurences in a document
                # Opening JSON file 
                f = open(file_path,)
                # returns JSON object as a dictionary 
                doc_data = json.load(f)
                for ind in json_selection[:-1]:
                    doc_data=doc_data[ind]
                text_lines_to_process=[]
                for data_part in doc_data:
                    text_lines_to_process.append((data_part[json_selection[-1][0]],data_part[json_selection[-1][1]]))
                #process text_lines_to_process
                for (text,cite_spans) in text_lines_to_process:
                    #text=fill_references(text=text,refs=cite_spans)
                    tokens=word_tokenize(text)
                    tokens=[strip_nonalnum_re(token.casefold()) for token in tokens]
                    tokens = list(filter(lambda token: (token not in string.punctuation) and (not token.isnumeric()), tokens))
                    for token in tokens:
                        docWordSet.add(token)
                for word in docWordSet:   #update wordDocfreq
                    if wordDocfreq.get(word) is None:
                        wordDocfreq[word]=1
                    else:
                        wordDocfreq[word]+=1
    for (key,value) in wordDocfreq.items():
        wordDocfreq[key]=math.log(tot_doc_count/wordDocfreq[key])
    return wordDocfreq
                    


        


# In[6]:


def save_dictionary(dict_path, dict_itself):
    # saves given dictionary to json with given name
    with open(dict_path+'.json', 'w') as file:
        json.dump(dict_itself, file, sort_keys=True, indent=4)

def save_matrix(path, matrix):
    # saves given dictionary to json with given name
    matrix_to_save=matrix.tolist()
    with open(path+'.json', 'w') as file:
        json.dump(matrix_to_save, file, indent=4)

def load_dictionary(dict_path):
    # loads dictionary from json with given name
    with open(dict_path+'.json', 'r') as file:
        return json.load(file)


# In[7]:



"""
given topic id, finds docsIDs and calculates cosine similarity matrix between documents
"""
def doc_graph(topic_id):
    docIds=relevant_docId_dict[topic_id]
    doc_count=len(docIds)
    cos_sim_matrix=-np.ones(shape=(doc_count,doc_count))
    for ind1 in range(doc_count):
        for ind2 in range(doc_count):
            if ind1==ind2:
                cos_sim_matrix[ind1,ind2]=1
                continue
            if cos_sim_matrix[ind2,ind1]!=-1:
                cos_sim_matrix[ind1,ind2]=cos_sim_matrix[ind2,ind1]
                continue
            abstract1=docId_abstract_dict[docIds[ind1]]
            abstract2=docId_abstract_dict[docIds[ind2]]
            cos_similarity= 1 - spatial.distance.cosine(vectorize(abstract1), vectorize(abstract2))
            if str(cos_similarity)=="nan":
                print("found nan")
                cos_similarity=0
            cos_sim_matrix[ind1,ind2]=cos_similarity
    return cos_sim_matrix,docIds

"""
given sentences and calculates cosine similarity matrix between sentences
"""
def sentence_graph(sentences):
    doc_count=len(sentences)
    cos_sim_matrix=-np.ones(shape=(doc_count,doc_count))
    for ind1 in range(doc_count):
        for ind2 in range(doc_count):
            if ind1==ind2:
                cos_sim_matrix[ind1,ind2]=1
                continue
            if cos_sim_matrix[ind2,ind1]!=-1:
                cos_sim_matrix[ind1,ind2]=cos_sim_matrix[ind2,ind1]
                continue
            sentence1=sentences[ind1]
            sentence2=sentences[ind2]
            cos_similarity= 1 - spatial.distance.cosine(vectorize(sentence1), vectorize(sentence2))
            if str(cos_similarity)=="nan":
                #print("found a sentence without any token")
                cos_similarity=0
            cos_sim_matrix[ind1,ind2]=cos_similarity
    return cos_sim_matrix

 


# In[8]:


"""
vectorize given text according to the word_index_dict and word_idf_dict
"""
def vectorize(text):
    vector=np.zeros(shape=(word_count))
    tokens=word_tokenize(str(text))
    for token in tokens:
        index=word_index_dict.get(token)
        if index is None:
            continue
        vector[index]+=word_idf_dict[token]
    return vector


# In[9]:


"""
get transition probability matrix for given adj_matrix according to teleportation_rate
"""

def get_t_prob_matrix(adj_matrix):
    edge_counts=np.sum(adj_matrix, axis = 1)
    element_count=adj_matrix.shape[0]
    t_prob_matrix=np.zeros(shape=(element_count,element_count))
    for i in range(element_count):
        for j in range(element_count):
            if adj_matrix[i,j]:
                t_prob_matrix[i,j]=(1-teleportation_rate)/edge_counts[i]
            t_prob_matrix[i,j]+=teleportation_rate/element_count
    
    return t_prob_matrix


"""
find stationary distribution of transition probability matrix with power method
"""
def calc_eigen_vector(t_prob_matrix):
    N=t_prob_matrix.shape[0]
    prob_init=1/N*np.ones(shape=(N))
    sigma=np.inf 
    while sigma>=error_tolerance:
        prob_new=np.dot(t_prob_matrix.T,prob_init)
        sigma=np.linalg.norm(prob_new-prob_init)
        prob_init=prob_new
    return prob_init


# In[10]:


"""
execution of algorithm with given topic_id
"""
def execute(topic_id):
    #creates document graph and calculates cosine similarity matrix
    cos_sim_matrix,docIds=doc_graph(topic_id)
    #save_matrix("cos_sim_matrix", cos_sim_matrix)
    adj_matrix = (cos_sim_matrix >= cos_sim_threshold).astype(int)
    t_prob_matrix=get_t_prob_matrix(adj_matrix)
    steady_distr=calc_eigen_vector(t_prob_matrix)
    #print(steady_distr)
    most_important_docs_indicies= np.argpartition(steady_distr, -10)[-10:]
    most_important_docIDs=[docIds[ind] for ind in most_important_docs_indicies]
    most_important_docs_PageRanks=[steady_distr[ind] for ind in most_important_docs_indicies]
    sentences=find_documents_sentences(most_important_docIDs)
    cos_sim_matrix_sentences=sentence_graph(sentences)
    adj_matrix_sentences = (cos_sim_matrix_sentences >= cos_sim_threshold).astype(int)
    steady_distr_sentences=calc_eigen_vector(get_t_prob_matrix(adj_matrix_sentences))
    most_important_sentences_indicies= np.argpartition(steady_distr_sentences, -20)[-20:]
    most_important_sentences=[sentences[ind] for ind in most_important_sentences_indicies]
    most_important_sentences_PageRanks=[steady_distr_sentences[ind] for ind in most_important_sentences_indicies]
    return [most_important_docIDs,most_important_docs_PageRanks,most_important_sentences,most_important_sentences_PageRanks]



# In[11]:


def print_results(results,topic_id):
    most_important_docIDs=results[0]
    most_important_docs_PageRanks=results[1]
    most_important_sentences=results[2]
    most_important_sentences_PageRanks=results[3]
    print("Top 10 DocIds for topic id : "+str(topic_id))
    print(most_important_docIDs)
    print("Top 10 Document's corresponding PageRanks for topic : "+str(topic_id))
    print(most_important_docs_PageRanks)
    print("Top 20 Sentence for topic : "+str(topic_id))
    print(most_important_sentences)
    print("Top 20 Sentence's corresponding PageRanks for topic : "+str(topic_id))
    print(most_important_sentences_PageRanks)


# In[12]:


###########---------------EXECUTION---------------############
##stores list of documentID's for given topicId
relevant_docId_dict=read_relevance_file()
#reads topicID's relevant documentIDs to given dict
docId_abstract_dict=find_documents_abstracts(relevant_docId_dict[1]+relevant_docId_dict[13]+relevant_docId_dict[17])
#stores source folder path and necessary selection parts to create idf dictionary
#docs_folder_path_and_selection=["2020-04-10/",["body_text",["text","cite_spans"]]]
#process docs
#word_idf_dict=read_process_docs(docs_folder_path_and_selection)
#save_dictionary(dict_path="word_idf_dict", dict_itself=word_idf_dict)
word_idf_dict=load_dictionary(dict_path="NecessaryFiles/word_idf_dict")
#map words to a integer for matrix indicies and store it in word_index_dict
word_count=len(word_idf_dict.keys())
word_index_dict={}
for index, word in enumerate(word_idf_dict.keys()) :
    word_index_dict[word]=index
# parameters for execution
cos_sim_threshold=0.1
teleportation_rate=0.15
error_tolerance=0.00001

#execute the algorithm with given topic_id
results=execute(topic_id=1)
print_results(results,topic_id=1)
#second topic
results=execute(topic_id=13)
print_results(results,topic_id=13)
#third topic
results=execute(topic_id=17)
print_results(results,topic_id=17)

