# -*- coding: utf-8 -*-
"""TeamprojectNLP.ipynb

Group Members:
    Vishal - 301169302
    Arpit - 
    Manpreet - 301175898
    
Importing libraries
"""

import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# import dataset
input_data = pd.read_csv('content/Youtube04-Eminem.csv')

"""2"""

# display top 5 rows of dataset
input_data.head()

# display last 5 rows of dataset
input_data.tail()

"""printing the shape of the data

"""

# display shape of dataset
input_data.shape

"""printing columns"""

# display column names
input_data.columns

"""checking the datatype"""

# display column datatypes
input_data.dtypes

"""checking for any missing value

"""

# check isnull values
input_data.isnull().isnull().sum()

"""6"""

# shuffle the dataset with fraction = 1
input_data = input_data.sample(frac=1)

# selecting the important feature
input_data = input_data[['CONTENT','CLASS']]

print(input_data.head())

# data preprocessiong

# conver the comments into lower case for better accuracy
input_data["CONTENT"] = input_data["CONTENT"].str.lower()
print(input_data['CONTENT'].head(10))

# remove stop words from the comments
input_data['CONTENT'] = input_data['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(input_data['CONTENT'].head(10))

# Lemmatization of comments 
input_data['CONTENT'] = input_data['CONTENT'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)]))
print(input_data['CONTENT'].head(10))

# selecting class column as a target(Y)
y = input_data['CLASS']

#  converting word into vectors because model cannot 
count_vector = CountVectorizer()
vectorizeddata = count_vector.fit_transform(input_data['CONTENT'])

"""4"""

# shape of vectorized dataset
print(vectorizeddata.shape)

"""5"""

# Create the tf-idf transformer 
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(vectorizeddata)

"""7"""

# split the dataset into test and train datasets
index = int(0.75 * len(y))
X_train, X_test = train_tfidf[:index], train_tfidf[index:]
y_train, y_test = y[:index], y[index:]

model = MultinomialNB()
model.fit(X_train,y_train)

# testing the model
y_pred=model.predict(X_test)

# Confussion Matrix
confusion_matrix(y_test,y_pred)

# Scoring functions
num_fold=7
accuracy_val= cross_val_score(model,X_train,y_train,scoring='accuracy',cv=num_fold)

# display k-folds accuracy
for i in range(len(accuracy_val)):
  print('Accuracy for k-fold '+str(i+1)+' is ',round(accuracy_val[i]*100,2),'%')

# accuracy of the model
print('Accuracy of the model is:',accuracy_val.mean()*100)

# testing of the model using custom inputs
comments = ['this song is very good','i love eminem', '+447935454150 lovely girl talk to me xxxï»¿', 
            'Alright ladies, if you like this song, then check out John Rage.Â  He&#39;s a smoking hot rapper coming into the game.Â  He&#39;s not better than Eminem lyrically, but he&#39;s hotter. Hear some of his songs on my channel.ï»¿', 
            'Rihanna and Eminem together are unstoppable.ï»¿']

test_data = count_vector.transform(comments)
test_data = tfidf.transform(test_data)

result = model.predict(test_data)

# predicted outputs using model
dist = {0:'ham',
        1:'spam'}
        
for output,comment in zip(result,comments):
  print(comment,' : ',dist[output])

