import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors

def task1():
    reviews = pd.read_excel("movie_reviews.xlsx")
    
    reviews["Sentiment"] = reviews["Sentiment"].map({"negative":0,"positive":1})
    reviews["Split"] = reviews["Split"].map({"train":0,"test":1}) 

    data = reviews[["Review","Sentiment"]]
    targetsplit = reviews["Split"]
    targetsent = reviews["Sentiment"]
    
    traindata = data[targetsplit==0]
    traintarget = traindata["Sentiment"]
    
    testdata = data[targetsplit==1]
    testtarget = traindata["Sentiment"]
    
    
    negativetrain = traindata[targetsent==0]
    positivetrain = traindata[targetsent==1]
    
    negativetest = testdata[targetsent==0]
    positivetest = testdata[targetsent==1]
    
    print("Positive reviews in train: ", len(positivetrain))
    print("Negative reviews in train: ", len(negativetrain))
    
    print("Positive reviews in test: ", len(positivetest))
    print("Negative reviews in test: ", len(negativetest))
    
    
    return traindata, traintarget, testdata, testtarget

def task2(traindata, minWordLength, minWordOccurence):
    reviews = traindata["Review"]
    
    reviews = reviews.str.replace('[^a-zA-Z0-9 ]', '').str.lower().str.split()
    
    wordOccurences = {}
    
    for subset in reviews:
        for word in subset:
            if (len(word)>=minWordLength):
                if (word in wordOccurences):
                    wordOccurences[word] = wordOccurences[word] + 1
                else:
                    wordOccurences[word]=1
    
    filteredWordOccurences = {}
    
    for word in wordOccurences:
        if wordOccurences[word]>=minWordOccurence:
            filteredWordOccurences[word] = wordOccurences[word]
            
    return filteredWordOccurences
        
    
def task3(traindata, traintarget, wordlength, wordocc):
    totalwordocc = task2(traindata, wordlength,wordocc)
    
    positivedata = traindata[traintarget==1]
    negativedata = traindata[traintarget==0]

    positivelist = reviewcount(positivedata, totalwordocc)
    negativelist = reviewcount(negativedata, totalwordocc) 
    
    return totalwordocc, positivelist, negativelist

def reviewcount(data, wordocc):
    reviews = data["Review"].str.replace('[^a-zA-Z0-9 ]', '').str.lower().str.split()
    returnlist = {}
    
    for review in reviews:
        for word in wordocc:
            if word in review:
                if word in returnlist:
                    returnlist[word] += 1
                else:
                    returnlist[word] = 1
                    
    for word in wordocc:
        if word not in returnlist:
            returnlist[word] = 0
    
                    
    return returnlist
       
def task4(totalwordocc, positivelist, negativelist, data):
    alpha = 1
    
    target = data["Sentiment"]
    negative = sum(target==0)
    positive = sum(target==1)
  
    positiveprob = {}
    negativeprob = {}
    
    for word in totalwordocc:
        #print(totalwordocc[word], positivelist[word], negativelist[word])
        likelihoodp = (positivelist[word] + alpha) / (sum(positivelist.values()) + 2*alpha)
        positiveprob[word] = likelihoodp
        
        likelihoodn = (negativelist[word] + alpha) / (sum(negativelist.values()) + 2*alpha)
        negativeprob[word] = likelihoodn
        
    priorpositive =  positive/len(data)
    priornegative = negative/len(data)

    return positiveprob, negativeprob, priorpositive, priornegative
    
def task5(instring, positiveprob, negativeprob, priorpos, priorneg):
    
    splitstring = instring.replace('[^a-zA-Z0-9 ]', '').lower().split()
    positivevalue = priorpos
    negativevalue = priorneg
     
    for word in splitstring:
        if word in positiveprob:
            positivevalue = positivevalue * positiveprob[word]
        if word in negativeprob:
            negativevalue = negativevalue * negativeprob[word]

    if positivevalue > negativevalue:
        return 1
    else:
        return 0
    ''' 
    positivevalue = 0
    negativevalue = 0
     
    for word in splitstring:
        if word in positiveprob:
            positivevalue = positivevalue + math.log(positiveprob[word])
        if word in negativeprob:
            negativevalue = negativevalue + math.log(negativeprob[word])
        
    likelihood_positive = math.exp(positivevalue)
    likelihood_negative = math.exp(negativevalue)
    
    if likelihood_positive/likelihood_negative > priorneg/priorpos:
        return 1
    else:
        return 0
   '''
   
  
def task6(traindata, traintarget):
    kf = model_selection.StratifiedKFold(n_splits=3)
    
    reviews = pd.read_excel("movie_reviews.xlsx")

    reviews["Sentiment"] = reviews["Sentiment"].map({"negative":0,"positive":1})
    reviews["Split"] = reviews["Split"].map({"train":0,"test":1})
    
    data = reviews[["Review","Split"]]
    target = reviews["Sentiment"]
    
    accuracy_list = {}
    
    for k in range(1,11):
        totalwordocc, positivelist, negativelist = task3(traindata, traintarget, k, 700)
        positiveprob, negativeprob, priorpos, priorneg = task4(totalwordocc, positivelist, negativelist, traindata)
    
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
    
        k_accuracy = []
        
        for train_index, test_index in kf.split(traindata, traintarget):
            predicted_labels = []
            for index in test_index:
                instring = data["Review"].iloc[index]
                predicted_label = task5(instring, positiveprob, negativeprob, priorpos, priorneg)
                predicted_labels.append(predicted_label)
            C = metrics.confusion_matrix(target[test_index], predicted_labels)
        
            accuracy = metrics.accuracy_score(target[test_index], predicted_labels)
            k_accuracy.append(accuracy)
            
            true_positives.append(C[0,0])
            true_negatives.append(C[1,1])            
            false_positives.append(C[1,0])
            false_negatives.append(C[0,1])
            print()
            print("True positives:", np.sum(true_positives))
            print("True negatives:", np.sum(true_negatives))
            print("False positives:", np.sum(false_positives))
            print("False negatives:", np.sum(false_negatives))
            print("k: ",k, "Accuracy", accuracy)
            print()
            
        mean_accuracy = 0
        for value in k_accuracy:
            mean_accuracy += value
        mean_accuracy = mean_accuracy/3
        accuracy_list[k] = mean_accuracy
        
    for k in accuracy_list:
        print("k:", k, " Mean Accuracy: ", accuracy_list[k])
           
    
    
def main():
    traindata, traintarget, testdata, testtarget = task1()
    #task5("excellent brilliant movie", positiveprob, negativeprob, priorpos, priorneg)
    task6(traindata, traintarget)
    
main()