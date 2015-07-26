
# coding: utf-8

# In[48]:

import pandas as pd
import nltk
from nltk.util import ngrams
nltk.download('punkt')
nltk.download('stopwords')

#Importing the dataset
get_ipython().magic('cd C:\\Users\\Matt\\Dropbox\\Python Workspace\\CROW\\CROL-PDF')
data = pd.read_csv("procPublicationRequest_Oct-Dec_2014_clean - procPublicationRequest_Oct-Dec_2014_clean.csv")

#Snagging the "human_readable" column
human_readableList = list(data['human_readable'])

#Turn the values into strings
strReadable = [str(a) for a in human_readableList]

#Split into individual words
listOfLists = [a.split() for a in strReadable]


# In[49]:

#While we're here, let's output the raw words to a text file
myCorpus = ''
for myEntry in strReadable:
    myCorpus = myCorpus + "\n"+ myEntry


# In[50]:

f = open('rawCorpus', 'w')
f.write(myCorpus)
f.close()


# In[51]:

#Now we have a data file that'll probably a little faster to mess with (maybe?)
file = open('rawCorpus.txt')
t = file.read()


# In[52]:

#Let's tokenize it and turn into an NLTK Text file
myCorpusTokenized = nltk.word_tokenize(t)


# In[53]:

corpusText = nltk.Text(myCorpusTokenized)


# In[54]:

#Now that we've got a bigger body of text, we can look at more interesting patterns in phrasing
corpusText.collocations()


# In[55]:

corpusFreqDist = nltk.FreqDist(corpusText)


# In[36]:

#Most commom words!
list(corpusFreqDist.most_common(50))


# In[56]:

#Let's clean things up a little bit.  Changing everything to lower-case is usually a good idea.  
#"Public Hearing" will equal "PUBLIC HEARING"
lowerTokens = [w.lower() for w in myCorpusTokenized]
lowerText = nltk.Text(lowerTokens)


# In[57]:

list(nltk.FreqDist(lowerText).most_common(50))
#Already saved some doubling-up!  Note the 5649 mentions of "the", instead of 4860 like in the last list


# In[58]:

#Let's see some bigrams!
corpusBigrams = list(ngrams(lowerTokens,2))


# In[59]:

corpusBigramFreqs = nltk.FreqDist(corpusBigrams)
corpusBigramFreqs.most_common(50)


# In[38]:

#Let's see tri-grams!
corpusTrigrams = list(ngrams(lowerTokens,3))
corpusTrigramFreqs = nltk.FreqDist(corpusTrigrams)
corpusTrigramFreqs.most_common(50)


# In[39]:

#We can keep going!
corpus4grams = list(ngrams(lowerTokens,4))
corpus4gramFreqs = nltk.FreqDist(corpus4grams)
corpus4gramFreqs.most_common(50)


# In[40]:

corpus5grams = list(ngrams(lowerTokens,5))
corpus5gramFreqs = nltk.FreqDist(corpus5grams)
corpus5gramFreqs.most_common(50)


# In[41]:

corpus6grams = list(ngrams(lowerTokens,6))
corpus6gramFreqs = nltk.FreqDist(corpus6grams)
corpus6gramFreqs.most_common(50)


# In[42]:

corpus7grams = list(ngrams(lowerTokens,7))
corpus7gramFreqs = nltk.FreqDist(corpus7grams)
corpus7gramFreqs.most_common(50)


# In[43]:

corpus8grams = list(ngrams(lowerTokens,8))
corpus8gramFreqs = nltk.FreqDist(corpus8grams)
corpus8gramFreqs.most_common(50)


# In[44]:

corpus9grams = list(ngrams(lowerTokens,9))
corpus9gramFreqs = nltk.FreqDist(corpus9grams)
corpus9gramFreqs.most_common(50)


# In[47]:

corpus5grams = ngrams(lowerTokens,5)
corpus5gramFreqs = nltk.FreqDist(corpus5grams)
corpus5gramFreqs.most_common(50)


# In[45]:

#...and let's stop here for now
corpus10grams = list(ngrams(lowerTokens,10))
corpus10gramFreqs = nltk.FreqDist(corpus10grams)
corpus10gramFreqs.most_common(50)


# In[ ]:



