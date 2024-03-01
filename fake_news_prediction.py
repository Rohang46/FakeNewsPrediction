import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
#printting the stop words in english
print(stopwords.words('english'))

#Data preporcessing

#loading the dataset to the pd dataframe 
news_dataset=pd.read_csv(r'D:\\train.csv')

print(news_dataset.shape)
#print first five raows of dataset
print(news_dataset.head())

#counting th number of missing value in dataset
print(news_dataset.isnull().sum())

#repalacing the null values with empty string
news_dataset=news_dataset.fillna('')

#meraging the authour name and title
news_dataset['content']=news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

#sperating the datas labale

X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']
print(X)
print(Y)

#STEAMING 
port_stem=PorterStemmer()
def steming(content):
    stemned_content=re.sub('[^a-zA-Z]',' ',content)
    stemned_content=stemned_content.lower()
    stemned_content=stemned_content.split()
    stemned_content=[port_stem.stem(word) for word in stemned_content if not word in stopwords.words('english')]
    stemned_content=' '.join(stemned_content)
    return stemned_content
news_dataset['content']=news_dataset['content'].apply(steming)
print(news_dataset['content'])

#sperating the data and label

X=news_dataset['content'].values
Y=news_dataset['label'].values
print(X)
print(Y)
Y.shape

#CONVERTING TEXT INTO NUMBER
vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)
print(X)

#SPLATING THE DATA
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#Trainnig the model
model=LogisticRegression()
model.fit(X_train,Y_train)

#Evaluation
#accuracy score on the traing the data
X_train_predition=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predition,Y_train)
print('Accuracay of the training the data:',training_data_accuracy)

#accuract score on the test data\
X_test_predition=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predition,Y_test)
print('Accuracay of the testing the data:',test_data_accuracy)

#MAKING PREDIACTIOVE SYSTEM

X_news=X_test[3]

prediction=model.predict(X_news)
print(prediction)

if(prediction[0]==0):
    print("The news is Reail")

else:
    print("The news is fake")

print(Y_test[3])


 