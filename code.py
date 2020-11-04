#!/usr/bin/env python
# coding: utf-8

# In[51]:


import csv
'''
#Converting text file to csv file.
with open('C:/Users/Hitesh Kumar/Desktop/House-votes-data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('C:/Users/Hitesh Kumar/Desktop/House-votes-data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('votes1','votes2','votes3','votes4','votes5','votes6','votes7','votes8','votes9','votes10','votes11','votes12','votes13','votes14','votesint(len(X))/4','votes16','type'))
        writer.writerows(lines)
'''

# In[52]:


import pandas as pd
dataset = pd.read_csv('House-votes-data.csv')   # importing the dataset
import warnings
warnings.filterwarnings("ignore")
import numpy as np     
import matplotlib.pyplot as plt     # for visualisations
import itertools     # for visualisations


# In[53]:


X = dataset.iloc[:, 1:-1].values          # independent variables (col 1-int(len(X))/4)
y = dataset.iloc[:,16].values             # dependent variable (col 16)

import collections
for i in range(0,len(X[0])):
    a = collections.Counter(X[:,i])       # to check the number of y's and n's and missing values('?') 
#     print(a)


# In[54]:


# pre-processing
# replacing '?' by most frequent value corresponding to the target value

for i in range(len(X[0])):
    repub_yes = 0
    repub_no = 0
    demo_yes = 0
    demo_no = 0
    for j in range(len(X)):
        if(X[:,i][j] == 'y' and y[:,][j] == 'republican'):
            repub_yes+=1
        elif(X[:,i][j] == 'n' and y[:,][j] == 'republican'):
            repub_no+=1
        elif(X[:,i][j] == 'y' and y[:,][j] == 'democrat'):
            demo_yes+=1
        elif(X[:,i][j] == 'n' and y[:,][j] == 'democrat'):
            demo_no+=1
    #print(repub_yes)
    if(repub_yes > repub_no):
        replace_re = 'y'
    else:
        replace_re = 'n'
    if(demo_yes > demo_no):
        replace_dem = 'y'
    else:
        replace_dem = 'n'
    for k in range(0,len(X)):
        if(X[:,i][k] == '?' and y[:,][k]=='republican'):
            X[:,i][k] = replace_re
        if(X[:,i][k] == '?' and y[:,][k]=='democrat'):
            X[:,i][k] = replace_dem


# In[55]:


# to check if the missing values are filled.

import collections
for i in range(0,len(X[0])):
    a = collections.Counter(X[:,i])


# In[56]:


# for converting qualitative data to numerical by replacing y's with 1 and n's with 0

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[57]:


for i in range(len(X[0])):
     X[:,i]= labelencoder_X.fit_transform(X[:,i])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[58]:


# randomly shuffling the entire dataset to ensure equality

import random

combined = list(zip(X, y))
random.shuffle(combined)
random.shuffle(combined)
random.shuffle(combined)


# In[59]:


acc = []             # for storing five fold accuracies   

y_test_large = []    # to store all the five test sets during 5-folds
y_pred_large = []    # to store all the five prediction sets during 5-folds

for fold in range(int(len(X))/4,len(X)+1,int(len(X))/4):
    train_data = combined[:fold-int(len(X))/4] + combined[fold:]
    test_data = combined[fold-int(len(X))/4:fold]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    x_train[:], y_train[:] = zip(*train_data)
    x_test[:], y_test[:] = zip(*test_data)
    
    
    # training
    
    no_of_rep = y_train.count(1)                   # we are treating republicans as 1 and democrats as 0
    no_of_dem = len(y_train) - no_of_rep

    p_rep = no_of_rep/len(y_train)
    p_dem = 1 - p_rep

    #print(p_rep, p_dem)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    d = {i:[0,0,0,0] for i in range(len(X[0]))}            # [(yes, republic), (no, republic), (yes, democrat), (no, democrat)]

    for column in range(int(len(X))/4):
        for item in range(len(y_train)):
            if(x_train[:,column][item] == 1 and y_train[:,][item] == 1):
                d[column][0] += 1
            elif(x_train[:,column][item] == 0 and y_train[:,][item] == 1):
                d[column][1] += 1
            elif(x_train[:,column][item] == 1 and y_train[:,][item] == 0):
                d[column][2] += 1
            elif(x_train[:,column][item] == 0 and y_train[:,][item] == 0):
                d[column][3] += 1

    #print(d)

    y_pred=[]
    count = 0
    true = 0
    false = 0
    
    #testing
    
    for item in x_test:
        temp = 0
        prob_rep = p_rep/(no_of_rep**int(len(X))/4)
        prob_dem = p_dem/(no_of_dem**int(len(X))/4)
        for i in item:
            if(i == 1):
                prob_rep *= d[temp][0]
                prob_dem *= d[temp][2]
            else:
                prob_rep *= d[temp][1]
                prob_dem *= d[temp][3]
            temp += 1
        
        # comparision
        if(prob_rep > prob_dem):
            pred = 1
        else:
            pred = 0
            
        y_pred.append(pred)
        
        if(pred == y_test[count]):
            true += 1
        else:
            false += 1
            
        count += 1
        
    y_test_large.append(y_test)
    y_pred_large.append(y_pred)
    t_acc = (true/(true+false))*100
    print("Accuracy = {}".format(t_acc))
    acc.append(t_acc)
#print(acc)
print("Average Accuarcy of five cross validation : {}".format(sum(acc)/5))


# In[60]:


# cm = confusion_matrix(y_test_large[0], y_pred_large[0])
# print(cm)


# In[61]:


from sklearn.metrics import confusion_matrix

for i in range(5):
    print("Confusion Matrix for validation no {} : {} ".format(i+1,confusion_matrix(y_test_large[i], y_pred_large[i])))
    tn, fp, fn, tp = confusion_matrix(y_test_large[i], y_pred_large[i]).ravel()
    Recall = tp/(tp+fn)
    Precision = tp/(tp+fp)
    F_Score = 2*(Precision*Recall)/(Precision+Recall)
    print("Precision = {} ".format(Precision*100))
    print("Recall = {} ".format(Recall*100))
    print("F-Score =  {}".format(F_Score*100))
    print()


# In[62]:


for i in range(5):
    cm = confusion_matrix(y_test_large[i], y_pred_large[i])

    plt.matshow(cm,cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix Plot')
    plt.colorbar()
    plt.xlabel('Precited')
    plt.ylabel('Actual')
    #plt.show()


    tick_marks = np.arange(len(set(y_pred))) # length of classes
    class_labels = ['Republic','Democratic']
    tick_marks
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),horizontalalignment='center',color='white' if cm[i,j] >thresh else 'black')
    plt.show()


# In[63]:


x = [1,2,3,4,5]
plt.plot(x, acc)
plt.xlim([0, 6])
plt.ylim([50, 100])
plt.show()

# In[64]:


for i in range(5):
    cm = confusion_matrix(y_test_large[i], y_pred_large[i])
    
    labels =   ['Republican - Republican', 'Republican-Democratic',  'Democratic-Democratic','Democratic-Republican']
    sizes = [cm[0][0],cm[0][1],cm[1][1],cm[1][0]]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0.05,0.05,0.05,0.05)

    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle) 
    plt.tight_layout()
    plt.title('Actual vs Predicted')

    plt.show()
