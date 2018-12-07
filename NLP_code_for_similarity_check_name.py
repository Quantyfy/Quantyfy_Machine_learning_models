# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:34:20 2018

@author: rbabu
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import time
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
import nameparser
from nameparser import HumanName

# Loading data
pd.set_option('display.max_colwidth', -1)
names =  pd.read_excel(r'C:\Users\rbabu\Desktop\Master Data.xlsx',encoding='cp1252')
print('The shape: %d x %d' % names.shape)
names.head()
#DOB Smilarity Check
df_dob=pd.DataFrame(names['DOB'])
df_dob['DOB']=df_dob['DOB'].astype(str)
df_dob['DOB'] = df_dob['DOB'].str.replace('000000', ' ')
df_dob['DOB'] = df_dob['DOB'].str.strip()

#N Grams
def ngrams(string, n=3):
    string =  re.sub("[^a-zA-Z]",
                          " ", 
                          str(string))
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]





def remove_punct(names):
    return re.sub('[^ A-Za-z0-9]+', '', str(names))

names['Name']=names['Name'].apply(lambda x:remove_punct(x))

df_dob['DOB']=df_dob['DOB'].apply(lambda x:remove_punct(x))

df_dob['DOB']=df_dob.drop(df_dob.index[1193])
df_dob=df_dob.dropna()
def preprocess(sentence):
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        filtered_words = [w for w in tokens if not w in stopwords.words('english')]
        return ' '.join(filtered_words)
    
names['Name']=names['Name'].apply(lambda x:preprocess(x))

re.split()

#Remove name titles
def remove_name_titles(name):
    name = HumanName(name)
    name.string_format = "{first} {last}"
    return str(name)
 
names['Name']=names['Name'].apply(lambda x:remove_name_titles(x))



Name = names['Name']
Names_1=names['Name']

names_1=Names_1.drop(1192)

Name.dtypes
df_dob.dtypes
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(Name)
print(tf_idf_matrix[0])

#DOB
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix_dob = vectorizer.fit_transform(df_dob)
print(tf_idf_matrix[0])

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix_1 = vectorizer.fit_transform(Names_1)
print(tf_idf_matrix_1[0])

#Names_1='anna ryan'
#
#Names_1=pd.Series(Names_1)


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


t1 = time.time()
matches = awesome_cossim_top(tf_idf_matrix_1, tf_idf_matrix.transpose(), 1423249, 0.3)
t = time.time()-t1
print("SELFTIMED:", t)

def get_matches_df(sparse_matrix, name_vector, top=2931270):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'Reading_string': left_side,
                          'Matched_string': right_side,
                           'similairity': similairity})

matches_df = get_matches_df(matches, Name, top=13716)
matches_df = matches_df[matches_df['similairity']<0.99999] # Remove all exact matches
matches_df.head(20)

matches_df.sort_values(['similairity'], ascending=False).head(10)

matches_df.to_csv(r'C:\Users\rbabu\Desktop\nlp_output.csv')