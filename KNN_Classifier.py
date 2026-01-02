# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 00:09:38 2025

@author: steph
"""
#K-Nearest Neighbors (KNN) Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv("mail_data.csv")

X = df['Message']
y= df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train) #this looks at training data only and converts text to numbers
X_test_vec = vectorizer.transform(X_test)  #prevents data leakage (prevents model from cheating)

knn = KNeighborsClassifier(n_neighbors = 3, metric="cosine")
knn.fit(X_train_vec, y_train)

y_predic = knn.predict(X_test_vec)

acc = accuracy_score(y_test, y_predic)
prec = precision_score(y_test, y_predic, pos_label='spam')    #precision 
rec = recall_score(y_test, y_predic, pos_label='spam') #without pos_label it would default to 1
f1 = f1_score(y_test, y_predic, pos_label='spam')

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1: {f1}")

x = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [acc, prec, rec, f1]

plt.figure(figsize=(5,5))
bars = plt.bar(x, values, color=['#99987f', '#b8b516', '#1315ba', '#ba1313'])

#this addsvalue labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("K-Nearest Neighbor Classifier Performance on Spam Detection")
plt.show()
