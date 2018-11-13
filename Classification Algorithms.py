
# coding: utf-8

# In[50]:


import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)



   # print(dictionary)
    
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
    
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix
    
    
# Create a dictionary of words with its frequency

train_dir = '/Users/cda/Desktop/Assignment1/ling-spam/train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare vectors for training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC()
model2 = MultinomialNB()
model3 = DecisionTreeClassifier()

model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)
model3.fit(train_matrix,train_labels)

# Test the unseen mails for Spam

test_dir = '/Users/cda/Desktop/Assignment1/ling-spam/test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model3.predict(test_matrix)

print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))
print(confusion_matrix(test_labels,result3))

