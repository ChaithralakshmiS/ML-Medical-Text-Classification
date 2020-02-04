#!/usr/bin/env python
# coding: utf-8

# In[76]:


#Import libraries
import numpy as np
import scipy as sp
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import defaultdict,Counter
import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import *
from nltk import PorterStemmer


# In[2]:


# Drop words wit length<minlength
def filterLen(docs, minlen):
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]


# In[ ]:





# In[3]:


# Build sparse matrix from a list of document
# each of which is a list of word/terms in the document.  

def build_matrix(docs):
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


# In[4]:


#Print out info about this CSR matrix. If non_empy, 
#report number of non-empty rows and cols as well
def csr_info(mat, name="", non_empy=False):

   if non_empy:
       print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
               name, mat.shape[0], 
               sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
               for i in range(mat.shape[0])), 
               mat.shape[1], len(np.unique(mat.indices)), 
               len(mat.data)))
   else:
       print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
               mat.shape[0], mat.shape[1], len(mat.data)) )


# In[5]:


# Scale a CSR matrix by idf. 
  # Returns scaling factors as dict. If copy is True, 
   # returns scaled matrix and scaling factors.
   
def csr_idf(mat, copy=False, **kargs):
  
   if copy is True:
       mat = mat.copy()
   nrows = mat.shape[0]
   nnz = mat.nnz
   ind, val, ptr = mat.indices, mat.data, mat.indptr
   # document frequency
   df = defaultdict(int)
   for i in ind:
       df[i] += 1
   # inverse document frequency
   for k,v in df.items():
       df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
   # scale by idf
   for i in range(0, nnz):
       val[i] *= df[ind[i]]
       
   return df if copy is False else mat


# In[6]:


# Normalize the rows of a CSR matrix by their L-2 norm. 
#   If copy is True, returns a copy of the normalized matrix.
#   
def csr_l2normalize(mat, copy=False, **kargs):
  
   if copy is True:
       mat = mat.copy()
   nrows = mat.shape[0]
   nnz = mat.nnz
   ind, val, ptr = mat.indices, mat.data, mat.indptr
   # normalize
   for i in range(nrows):
       rsum = 0.0    
       for j in range(ptr[i], ptr[i+1]):
           rsum += val[j]**2
       if rsum == 0.0:
           continue  # do not normalize empty rows
       rsum = 1.0/np.sqrt(rsum)
       for j in range(ptr[i], ptr[i+1]):
           val[j] *= rsum
           
   if copy is True:
       return mat


# In[7]:


#get_ipython().run_cell_magic('latex', '', '$$cos(\\mathbf{a}, \\mathbf{b}) = \\frac{\\langle \\mathbf{a}, \n          \\mathbf{b} \\rangle}{||\\mathbf{a}||\\ ||\\mathbf{b}||}$$')


# In[81]:


def cleanData(docs):
    d2=docs

    for i in range (0,len(docs)):
        for j in range(0,len(docs[i])):
            docs[i][j]=docs[i][j].lower()
            docs[i][j]=docs[i][j].translate(str.maketrans('', '', string.punctuation))
            regex = re.compile('[^a-zA-Z]')
            docs[i][j]=regex.sub('', docs[i][j])
            docs[i][j]=PorterStemmer().stem(docs[i][j])


        docs[i] = [ x for x in docs[i] if x is not '' ]
        

        #docs[i]=docs[i].remove('')

  
    return docs


# In[82]:


# Read train and test data
with open("train.dat", "r") as fh:
    lines_train = fh.readlines()
#print(len(lines_train))

with open("test.dat", "r") as fh:
    lines_test = fh.readlines()

#Combine train and test data
lines=lines_train+lines_test
#print('No of training data : ',len(lines_train))
#print('No of test data : ',len(lines_test))
#print('No of Toatal data : ',len(lines))
len_train= len(lines_train)
len_test = len(lines_test)
len_all=len(lines)
#print(len_train,len_test,len_all)


docs = [l.split() for l in lines]
docs1 = filterLen(docs, 4)
docs1 = cleanData(docs1)
print(docs1[0],docs1[10])
docs_train = [l.split() for l in lines_train]
output_class = np.zeros(len(lines_train),dtype=np.float)
for i in range(0, len(lines_train)):
        output_class[i] = float(docs_train[i][0])
        docs_train[i][0]=''
    

mat = build_matrix(docs)
mat1 = build_matrix(docs1)
csr_info(mat)
csr_info(mat1)
mat2 = csr_idf(mat1, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)


# In[ ]:



mat3_train=mat3[:len_train,:]
mat3_test=mat3[len_train:,:]
dots = mat3_test.dot(mat3_train.T)
print(dots.shape)


# In[ ]:


def kNearestNeighbors(k=6):
    for i in range(len_test):
        sims=list(zip(dots[i].indices,dots[i].data))
        sims.sort(key=lambda sims: sims[1], reverse=True)
        outcls=[output_class[s[0]] for s in sims[:k] if s[1]>0]
        maxCount=0
        classOutput=0
        count=[0,0,0,0,0,0]
        class_test=[]
        for i in range(k):
            count[int(outcls[i])]=count[int(outcls[i])]+1
        max_index=count.index(max(count))
        file.writelines(str(count[max_index])+'\n') 
        #print(count[max_index],'\n')


            #print(count,max_index,count[max_index])
file = open('format2.dat','w') 

kNearestNeighbors()
file.close()


# In[ ]:




