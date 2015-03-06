
# coding: utf-8

# In[4]:

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


# In[5]:

#Let's see an entry!
strReadable[0]
#It's a giant mess of text!


# In[21]:

#Here's what happens when we tokenize with NLTK
firstEntryTokenized = nltk.word_tokenize(strReadable[0])
#It splits it into a list of individual words
firstEntryTokenized[:50]


# In[9]:

#We can use some special functions if we then convert this into NLTK's special Text format
firstEntryText = nltk.Text(firstEntryTokenized)


# In[10]:

#For instance, we can get a list of the most common words along with how often they show up
firstEntryFreqDist = nltk.FreqDist(firstEntryText)


# In[14]:

#Let's see the 10 most common words!
firstEntryFreqDist.most_common(10)


# In[24]:

#We can also search for phrases of different lengths
firstEntryBigrams = list(ngrams(firstEntryTokenized,2))
firstEntryBigrams[:50]


# In[23]:

#And frequencies of phrases!
firstEntryFreq= nltk.FreqDist(firstEntryBigrams)
firstEntryFreq.most_common(50)


# In[ ]:



