
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:37:27 2025

@author: seanmanares
"""
#Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

df = pd.read_csv("mail_data.csv")
print(df.head)
print(df.shape)

X = df['Message']
y = df['Category']   #ham or spam

vectorizer = CountVectorizer(stop_words='english') #this converts text to numbers 
X_numeric = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_predic = log_reg.predict(X_test)
acc = accuracy_score(y_test, y_predic)   #accuracy
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
plt.title("Logistic Regression Performance on Spam Detection")
plt.show()



