# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import re
import random
import numpy
import matplotlib.pyplot as grp
import math
import time
from random import randrange
from math import sqrt

class feature:
    def __init__(self, word, freq):
                self.word = word
                self.freq = freq
    def __str__(self):
        return "%s %d" % (self.word, self.freq)


def read_file(file):
    dictionaryForm = {}
    with open(file) as f:           
        for row in f:
            token=row.split('\t')       
            reviewToken=re.sub(r'[^a-zA-Z0-9\s]','',token[2]).strip().split()  
            docId=token[0]
            label=token[1]
            for i in range(len(reviewToken)):
                reviewToken[i]=reviewToken[i].strip().lower()
                reviewToken[i]=re.sub(r'\W+','',reviewToken[i])     
            dictionaryForm[docId]=(label,reviewToken)
    
    return dictionaryForm

if(len(sys.argv) == 4):   
    #print(type(sys.argv[1]))       
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]
    modelIdx = sys.argv[3]              # DT=1, Bagging=2, RF=3
        
    dictTrainFile = read_file(trainFileName)
    dictTestFile = read_file(testFileName)
    
    def construct_features(dictTrainFile,n_features): 
        ##count unique no of words in each review
        featureWords=[]
        featureFreq=[]
        featureList=[]
        for key in dictTrainFile:
            eachReviewSet = set(dictTrainFile[key][1])       #this is a review text
            for setValue in eachReviewSet:                  #setValue : each word in review
                if (setValue in featureWords):
                    ##increment freq
                    wordIndex=featureWords.index(setValue)
                    featureFreq[wordIndex]= featureFreq[wordIndex] + 1
                else:
                    featureWords.append(setValue)
                    wordIndex=featureWords.index(setValue)
                    featureFreq.insert(wordIndex,1)
        for i in range(len(featureWords)):
            featureList.append(feature(featureWords[i],featureFreq[i]))
        featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
        featureListNew = featureList[100:]
        ##consider features from 100 to 1000
        if(len(featureListNew)>=n_features):
            featureListNew = featureListNew[0:n_features]
        else:
            featureListNew = featureListNew[0:]
        #print("25th index: ",featureListNew[25].word)
        return featureListNew
    
    def construct_train_f_vectors(dictTrainFile,featureListNew):
        ##now,construct 1000-dimensional vector for each review!
        vectorTupleTrain={}
        for key in dictTrainFile:
            eachVectorListTrain=[]
            eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
            for i in range(len(featureListNew)):
                if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                    eachVectorListTrain.append(1)
                else:
                    eachVectorListTrain.append(0)
            vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
        return vectorTupleTrain
        
    def construct_test_f_vectors(dictTestFile,featureListNew):
        ##next,forming vector tuples for test set as well
        vectorTupleTest={}
        for key in dictTestFile:
            eachVectorListTest=[]
            eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
            for i in range(len(featureListNew)):
                if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                    eachVectorListTest.append(1)
                else:
                    eachVectorListTest.append(0)
            vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
        return vectorTupleTest
    
    ##########
    def split_gini(f_index, dataset):
        countX0C0=0
        countX0C1=0
        countX1C0=0
        countX1C1=0
        left_child={}
        right_child={}
        for key in dataset:  #take analogy: dataset is dictionary of vector tuple (docid:label,featureVector)
           if(dataset[key][1][f_index]==0):
                left_child[key]=dataset[key] #left_child.append(dataset[key])         #child node is list of vector tuples
                if(dataset[key][0]=='0'):
                    countX0C0 = countX0C0 + 1
                else:
                    countX0C1 = countX0C1 + 1
           else:
                right_child[key]=dataset[key]
                #right_child.append(dataset[key])  
                if(dataset[key][0]=='0'):
                    countX1C0 = countX1C0 + 1
                else:
                    countX1C1 = countX1C1 + 1
        gini_value=0.0
        gini_gain=0.0
        parent_len=len(dataset)
        left_child_len=len(left_child)
        right_child_len=len(right_child)
        if(left_child_len==0):
            gini_value=0.0                                  #??????
        else:
            px=0.0
            px=math.pow((countX0C0/left_child_len),2)
            px=px+math.pow((countX0C1/left_child_len),2)
            px=1-px
            gini_value=(left_child_len/float(parent_len))*px
        if(right_child_len==0):
            gini_value = 0.0 + gini_value
        else:
            px=0.0
            px=math.pow((countX1C0/right_child_len),2)
            px=px+math.pow((countX1C1/right_child_len),2)
            px=1-px
            gini_value=gini_value + (right_child_len/float(parent_len))*px
        
        countPC0=countX1C0+countX0C0
        countPC1=countX1C1+countX0C1
        px=0.0
        px=math.pow((countPC0/parent_len),2)
        px=px+math.pow((countPC1/parent_len),2)
        px=1-px
        gini_gain=float(px-gini_value)
        return gini_gain, left_child, right_child
    #end of split_gini
    
    def boost_split_gini(f_index,dataset):
        countX0C0=0.0
        countX0C1=0.0
        countX1C0=0.0
        countX1C1=0.0
    
        left_child={}
        right_child={}
        for key in dataset:  #take analogy: dataset is dictionary of vector tuple (docid:label,featureVector)
           if(dataset[key][1][f_index]==0):
                left_child[key]=dataset[key] #left_child.append(dataset[key])         #child node is list of vector tuples
                if(dataset[key][0]=='0'):
                    countX0C0 = countX0C0 + dataset[key][2]
                else:
                    countX0C1 = countX0C1 + dataset[key][2]
           else:
                right_child[key]=dataset[key]
                #right_child.append(dataset[key])  
                if(dataset[key][0]=='0'):
                    countX1C0 = countX1C0 + dataset[key][2]
                else:
                    countX1C1 = countX1C1 + dataset[key][2]
        gini_value=0.0
        gini_gain=0.0
        countPC0=countX1C0+countX0C0
        countPC1=countX1C1+countX0C1
        #parent_len=len(dataset)
        left_child_len=len(left_child)
        right_child_len=len(right_child)
        if(left_child_len==0):
            gini_value=0.0                                  
        else:
            px=0.0
            px=math.pow((countX0C0/(countX0C1+countX0C0)),2)
            px=px+math.pow((countX0C1/(countX0C1+countX0C0)),2)
            px=1-px
            gini_value=((countX0C1+countX0C0)/(countPC0+countPC1))*px
                       
        if(right_child_len==0):
            gini_value = 0.0 + gini_value
        else:
            px=0.0
            px=math.pow((countX1C0/(countX1C0+countX1C1)),2)
            px=px+math.pow((countX1C1/(countX1C0+countX1C1)),2)
            px=1-px
            gini_value=gini_value + (((countX1C0+countX1C1)/(countPC0+countPC1))*px)
        
        
        px=0.0
        px=math.pow((countPC0/(countPC0+countPC1)),2)
        px=px+math.pow((countPC1/(countPC0+countPC1)),2)
        px=1-px
        gini_gain=float(px-gini_value)
#        print("index: ",f_index," :gain: ",gini_gain)
        return gini_gain, left_child, right_child
    #end of boost_split_gini
    
    def best_split(dataset,rf,no_of_features,featureListNew,boost):
        best_gini_gain=0.0
        if(rf==True):
            reduced_features=[]
            while(len(reduced_features)<no_of_features):
                index = randrange(len(featureListNew))
                if index not in reduced_features:
                    reduced_features.append(index)
            features_selected=list(reduced_features)
        else:
            features_selected=list(range(len(featureListNew)))
        for f_index in features_selected:
            if(boost==True):
                gini_gain, left_child1, right_child1 = boost_split_gini(f_index,dataset)
                #print("index: ",f_index," :gain: ",gini_gain)
            else:
                gini_gain, left_child1, right_child1 = split_gini(f_index,dataset)
            if(gini_gain>=best_gini_gain):
                best_f_index=f_index
                best_gini_gain=gini_gain
                left_child={}
                right_child={}
                left_child=left_child1.copy()
                right_child=right_child1.copy()
        #print("best feature index: ",best_f_index," :gini gain: ",best_gini_gain)
        return {'best_index':best_f_index, 'left':left_child,'right':right_child}
    #end of best_split
    
    def create_leaf(dataset,boost):
        if(boost==True):
            sum_weights=0
            for key in dataset:
                if(int(dataset[key][0])==0):
                    actual=-1
                else:
                    actual=1
                sum_weights=sum_weights+(dataset[key][2]*actual)
            if(sum_weights>0):
                return 1
            else:
                return 0
        else:
            class_labels=[]
            for key in dataset:
                class_labels.append(int(dataset[key][0]))
            return max(set(class_labels), key=class_labels.count)
    #end of create_leaf
    
    def rec_split(curr_node,max_depth,min_examples,depth,rf,no_of_features,featureListNew,boost):
        #if all instances in one child
        if not curr_node['left']:
            curr_node['left'] = curr_node['right'] = create_leaf(curr_node['right'],boost)
            #print("prediction all right: ",curr_node['left'])
            return
        if not curr_node['right']:
            curr_node['left'] = curr_node['right'] = create_leaf(curr_node['left'],boost)
            #print("prediction all left: ",curr_node['left'])
            return
        #if max depth
        if depth >= max_depth:
            curr_node['left'] = create_leaf(curr_node['left'],boost)
            #print("prediction left: ",curr_node['left'])
            curr_node['right'] = create_leaf(curr_node['right'],boost)
            #print("prediction right: ",curr_node['right'])
            return
        #recursion on left child
        if(boost==True):
            sum_w=0
            for key in curr_node['left']:
                sum_w=sum_w+curr_node['left'][key][2]
            #stop_cr=len(curr_node['left'])*sum_w
            if(sum_w < min_examples):
                curr_node['left'] = create_leaf(curr_node['left'],boost)
            else:    
                curr_node['left'] = best_split(curr_node['left'],rf,no_of_features,featureListNew,boost)
                rec_split(curr_node['left'], max_depth, min_examples, depth+1,rf,no_of_features,featureListNew,boost)  
        elif(len(curr_node['left']) < min_examples):
            curr_node['left'] = create_leaf(curr_node['left'],boost)
        else:
            curr_node['left'] = best_split(curr_node['left'],rf,no_of_features,featureListNew,boost)
            rec_split(curr_node['left'], max_depth, min_examples, depth+1,rf,no_of_features,featureListNew,boost)
        #recursion on right child
        if(boost==True):
            sum_w=0
            for key in curr_node['right']:
                sum_w=sum_w+curr_node['right'][key][2]
            #stop_cr=len(curr_node['right'])*sum_w
            if(sum_w < min_examples):
                curr_node['right'] = create_leaf(curr_node['right'],boost)
            else:
                curr_node['right'] = best_split(curr_node['right'],rf,no_of_features,featureListNew,boost)
                rec_split(curr_node['right'], max_depth, min_examples, depth+1,rf,no_of_features,featureListNew,boost)
        elif(len(curr_node['right']) < min_examples):
            curr_node['right'] = create_leaf(curr_node['right'],boost)
        else:
            curr_node['right'] = best_split(curr_node['right'],rf,no_of_features,featureListNew,boost)
            rec_split(curr_node['right'], max_depth, min_examples, depth+1,rf,no_of_features,featureListNew,boost)
    #end of rec_split

    def build_tree(train,max_depth,min_examples,rf,no_of_features,featureListNew,boost):
        root_node = best_split(train,rf,no_of_features,featureListNew,boost)
        rec_split(root_node, max_depth,min_examples,1,rf,no_of_features,featureListNew,boost)
        return root_node
    #end of build_tree
    
    def predict(curr_node,each_instance):
        if(each_instance[1][curr_node['best_index']]==0):   #move to left branch
            if isinstance(curr_node['left'], dict):
                return predict(curr_node['left'], each_instance)
            else:
                return curr_node['left']
        else:
            if isinstance(curr_node['right'], dict):
                return predict(curr_node['right'], each_instance)
            else:
                return curr_node['right']
    ##########
    #bagging related
    def sample_rep(dataset):
        arrList=[]
        while(len(arrList)<len(dataset)):
            random_key=random.choice(list(dataset))
            arrList.append(dataset[random_key])
        sample={}
        for i in range(len(arrList)):
            sample[i]=arrList[i]
        return sample
    #end of sample_rep
    
    def bagging_predict(tree_node_list,each_instance):
        prediction_list=[]
        for tree_node in tree_node_list:
            each_prediction=predict(tree_node,each_instance)
            prediction_list.append(each_prediction)
        return max(set(prediction_list), key=prediction_list.count)
    
    def boosting_predict(tree_node_list, each_instance):
        #print("boosting instance: ",each_instance)
        sum_weighted_pred=0.0
        for tree_node in tree_node_list:
            each_prediction=predict(tree_node,each_instance)
            #print("prediction: ",each_prediction,"each_instance: ",each_instance)
            if(each_prediction==0):
                each_prediction=-1
            sum_weighted_pred=sum_weighted_pred+(tree_node['alpha']*each_prediction)
            #print("sum_weighted_pred: ",sum_weighted_pred)
        if(sum_weighted_pred>0):
            #print("final pred: 1")
            return 1
        else:
            #print("final pred: 0")
            return 0
    ##########
    
    #learn DT model
    if(modelIdx=='1'):  
        #start_time = time.time()
        n_features=1000
        featureListNew=construct_features(dictTrainFile,n_features)
        vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
        vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
        
        train_data=vectorTupleTrain
        test_data=vectorTupleTest
        #print(test_data)
        max_depth=10
        min_examples=10

        misclassify=0
        totalClassify=len(test_data)
        tree_node = build_tree(train_data, max_depth, min_examples,False,n_features,featureListNew,False)
        for key in test_data:
            prediction = predict(tree_node, test_data[key])
            if(prediction!=int(test_data[key][0])):
                misclassify= misclassify +1
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-DT ",zeroOneLoss)
        #print("--- %s seconds ---" % (time.time() - start_time))
        
    #learn Bagging model
    if(modelIdx=='2'):
        #start_time = time.time()
        n_features=1000
        featureListNew=construct_features(dictTrainFile,n_features)
        vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
        vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
        
        train_data=vectorTupleTrain
        test_data=vectorTupleTest
        max_depth=10
        min_size=10
        number_trees=50

        tree_node_list=[]
        misclassify=0
        totalClassify=len(test_data)
        for i in range(number_trees):
            sample_dataset=sample_rep(vectorTupleTrain)
            tree_node=build_tree(sample_dataset,max_depth, min_size,False,n_features,featureListNew,False)
            tree_node_list.append(tree_node)
        for key in test_data:
            prediction = bagging_predict(tree_node_list, test_data[key])
            if(prediction!=int(test_data[key][0])):
                misclassify= misclassify +1
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-BT ",zeroOneLoss)
        #print("--- %s seconds ---" % (time.time() - start_time))
    
    #learn RF model
    if(modelIdx=='3'):
        #start_time = time.time()
        n_features=1000
        featureListNew=construct_features(dictTrainFile,n_features)
        vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
        vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
        
        train_data=vectorTupleTrain
        test_data=vectorTupleTest
        max_depth=10
        min_size=10
        number_trees=50

        no_of_features=int(sqrt(len(featureListNew)))
        tree_node_list=[]
        misclassify=0
        totalClassify=len(test_data)
        for i in range(number_trees):
            sample_dataset=sample_rep(vectorTupleTrain)
            tree_node=build_tree(sample_dataset,max_depth, min_size,True,no_of_features,featureListNew,False)
            tree_node_list.append(tree_node)
        for key in test_data:
            prediction = bagging_predict(tree_node_list, test_data[key])
            if(prediction!=int(test_data[key][0])):
                misclassify= misclassify +1
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-RF ",zeroOneLoss)
        #print("--- %s seconds ---" % (time.time() - start_time))
        
    #learn Boosting model
    if(modelIdx=='4'):
        #start_time = time.time()
        n_features=1000
        featureListNew=construct_features(dictTrainFile,n_features)
        vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
        vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
        
        train_data=vectorTupleTrain
        test_data=vectorTupleTest
        max_depth=10
        min_size=10
        min_size=min_size/len(train_data)
        number_trees=50

        tree_node_list=[]
        misclassify=0
        totalClassify=len(test_data)
        for key in train_data:
            train_data[key]=train_data[key]+(1/len(train_data),)
        
        for i in range(number_trees):
            count=0
            weighted_error=0
            sum_new_weight=0
            tree_node=build_tree(train_data,max_depth, min_size,False,n_features,featureListNew,True)
            
            for key in train_data:
                prediction = predict(tree_node, train_data[key])
                if(prediction==0):
                    prediction=-1
                if(train_data[key][0]=='0'):
                    actual=-1
                else:
                    actual=1
                if(prediction!=actual):             #misclassified training event
                    weighted_error=weighted_error+train_data[key][2]
                    count=count+1
                if(len(train_data[key])<5):
                    train_data[key]=train_data[key]+(prediction,actual)
                else:
                    lst=[]
                    lst=list(train_data[key])
                    lst[3]=prediction
                    lst[4]=actual
                    train_data[key]=tuple(lst)
            #print("round: ",i,"train_data: ",train_data)
            #generated err
            #print("round: ",i," weighted error: ",weighted_error)
            #print("round: ",i," misclassified events: ",count)
            if(weighted_error==0):
                log=math.log((1+sys.float_info.epsilon)/(sys.float_info.epsilon))   #?????? 10 -6
            else:
                log=math.log((1-weighted_error)/weighted_error)
            alpha=(0.5)*log
            tree_node['alpha']=alpha
            #print("round: ",i," :alpha: ",alpha)
            tree_node_list.append(tree_node)
            for key in train_data:
                lst=[]
                lst=list(train_data[key])
                lst[2]=(train_data[key][2])*(math.exp(-(alpha*train_data[key][3]*train_data[key][4])))
                train_data[key]=tuple(lst)
                #train_data[key][2]=(train_data[key][2])*(alpha*train_data[key][3]*train_data[key][4])
                sum_new_weight=sum_new_weight+train_data[key][2]
            #renormalize again
            for key in train_data:
                lst=[]
                lst=list(train_data[key])
                lst[2]=train_data[key][2]/sum_new_weight
                train_data[key]=tuple(lst)
                #train_data[key][2]=train_data[key][2]/sum_new_weight
            #print("round: ",i,"train_data: ",train_data)
        #complete all trees
        for key in test_data:
            prediction = boosting_predict(tree_node_list, test_data[key])
            if(prediction!=int(test_data[key][0])):
                misclassify= misclassify +1
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-BST ",zeroOneLoss)
        #print("--- %s seconds ---" % (time.time() - start_time))
    #end of boosting
              
    ques=0
    if(ques==1):
        ##INCREMENTAL partition
        ##different training sizes
        #print("start analysis 1")
        DTavgZeroOneLossList=[]
        DTstdZeroOneLossList=[]
        BTavgZeroOneLossList=[]
        BTstdZeroOneLossList=[]
        RFavgZeroOneLossList=[]
        RFstdZeroOneLossList=[]
        BSTavgZeroOneLossList=[]
        BSTstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        
        per=[0.025, 0.05, 0.125, 0.25]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for perc in per:
            #print("for perc: ",perc)
            DTZeroOneLossList=[]
            BTZeroOneLossList=[]
            RFZeroOneLossList=[]
            BSTZeroOneLossList=[]
            SVMZeroOneLossList=[]
            
            for trial in range(10):
                #print("for trial: ",trial)
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                n_features=1000
                featureListNew=construct_features(dictTrainFile,n_features)
                vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
                vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
                
                train_data=vectorTupleTrain
                test_data=vectorTupleTest
                max_depth=10
                min_size=10
                
                #learn DT
                #print("learning DT")
                misclassify=0
                totalClassify=len(test_data)
                tree_node = build_tree(train_data, max_depth, min_size,False,n_features,featureListNew,False)
                for key in test_data:
                    prediction = predict(tree_node, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("Dt loss: ",zeroOneLoss)
                DTZeroOneLossList.append(zeroOneLoss)
                #end DT
                
                #learn BT
                #print("learning BT")
                number_trees=50
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,False,n_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("Bt loss: ",zeroOneLoss)
                BTZeroOneLossList.append(zeroOneLoss)
                #end BT
                
                #learn RF
                #print("learning RF")
                number_trees=50
                no_of_features=int(sqrt(len(featureListNew)))
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,True,no_of_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("RF loss: ",zeroOneLoss)
                RFZeroOneLossList.append(zeroOneLoss)
                #end RF
                
                #learn BST
                #print("learning BST")
                min_size=min_size/len(train_data)
                number_trees=50
        
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for key in train_data:
                    train_data[key]=train_data[key]+(1/len(train_data),)
                
                for i in range(number_trees):
                    weighted_error=0
                    sum_new_weight=0
                    tree_node=build_tree(train_data,max_depth, min_size,False,n_features,featureListNew,True)
                    
                    for key in train_data:
                        prediction = predict(tree_node, train_data[key])
                        if(prediction==0):
                            prediction=-1
                        if(train_data[key][0]=='0'):
                            actual=-1
                        else:
                            actual=1
                        if(prediction!=actual):             #misclassified training event
                            weighted_error=weighted_error+train_data[key][2]
                        if(len(train_data[key])<5):
                            train_data[key]=train_data[key]+(prediction,actual)
                        else:
                            lst=[]
                            lst=list(train_data[key])
                            lst[3]=prediction
                            lst[4]=actual
                            train_data[key]=tuple(lst)
                    #print("round: ",i,"train_data: ",train_data)
                    #generated err
                    if(weighted_error==0):
                        log=math.log((1+sys.float_info.epsilon)/(sys.float_info.epsilon))
                    else:
                        log=math.log((1-weighted_error)/weighted_error)
                    alpha=(0.5)*log
                    tree_node['alpha']=alpha
                    #print("round: ",i," :alpha: ",alpha)
                    tree_node_list.append(tree_node)
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=(train_data[key][2])*(math.exp(-(alpha*train_data[key][3]*train_data[key][4])))
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=(train_data[key][2])*(alpha*train_data[key][3]*train_data[key][4])
                        sum_new_weight=sum_new_weight+train_data[key][2]
                    #renormalize again
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=train_data[key][2]/sum_new_weight
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=train_data[key][2]/sum_new_weight
                    #print("round: ",i,"train_data: ",train_data)
                #complete all trees
                for key in test_data:
                    prediction = boosting_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BST loss: ",zeroOneLoss)
                BSTZeroOneLossList.append(zeroOneLoss)
                #end BST
                                
                #start SVM
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                #print("SVM loss: ",zeroOneLoss)
                SVMZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            DTavgZeroOneLoss=numpy.average(DTZeroOneLossList)
            DTstdZeroOneLoss=numpy.std(DTZeroOneLossList)/math.sqrt(10)
            DTavgZeroOneLossList.append(DTavgZeroOneLoss)
            DTstdZeroOneLossList.append(DTstdZeroOneLoss)
            BTavgZeroOneLoss=numpy.average(BTZeroOneLossList)
            BTstdZeroOneLoss=numpy.std(BTZeroOneLossList)/math.sqrt(10)
            BTavgZeroOneLossList.append(BTavgZeroOneLoss)
            BTstdZeroOneLossList.append(BTstdZeroOneLoss)
            RFavgZeroOneLoss=numpy.average(RFZeroOneLossList)
            RFstdZeroOneLoss=numpy.std(RFZeroOneLossList)/math.sqrt(10)
            RFavgZeroOneLossList.append(RFavgZeroOneLoss)
            RFstdZeroOneLossList.append(RFstdZeroOneLoss)
            BSTavgZeroOneLoss=numpy.average(BSTZeroOneLossList)
            BSTstdZeroOneLoss=numpy.std(BSTZeroOneLossList)/math.sqrt(10)
            BSTavgZeroOneLossList.append(BSTavgZeroOneLoss)
            BSTstdZeroOneLossList.append(BSTstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            
        #after perc list
        print("DTavgZeroOneLossList : ",DTavgZeroOneLossList)
        print("DTstdZeroOneLossList : ",DTstdZeroOneLossList)
        print("BTavgZeroOneLossList : ",BTavgZeroOneLossList)
        print("BTstdZeroOneLossList : ",BTstdZeroOneLossList)
        print("RFavgZeroOneLossList : ",RFavgZeroOneLossList)
        print("RFstdZeroOneLossList : ",RFstdZeroOneLossList)
        print("BSTavgZeroOneLossList : ",BSTavgZeroOneLossList)
        print("BSTstdZeroOneLossList : ",BSTstdZeroOneLossList)
        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
        
        grp.figure(1)
        grp.errorbar(per, DTavgZeroOneLossList, DTstdZeroOneLossList,  marker='^',  label = "DT 0-1 loss")
        grp.errorbar(per, BTavgZeroOneLossList, BTstdZeroOneLossList,  marker='^',  label = "BT 0-1 loss")
        grp.errorbar(per, RFavgZeroOneLossList, RFstdZeroOneLossList,  marker='^',  label = "RF 0-1 loss")
        grp.errorbar(per, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.errorbar(per, BSTavgZeroOneLossList, BSTstdZeroOneLossList,  marker='^',  label = "BST 0-1 loss")
        grp.xlabel('Training set size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('training_size_loss_q1.png')
        
    elif(ques==2):
        ##INCREMENTAL partition
        ##different features
        #print("start analysis 2")
        DTavgZeroOneLossList=[]
        DTstdZeroOneLossList=[]
        BTavgZeroOneLossList=[]
        BTstdZeroOneLossList=[]
        RFavgZeroOneLossList=[]
        RFstdZeroOneLossList=[]
        BSTavgZeroOneLossList=[]
        BSTstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        
        perc=0.25
        n_fL=[200, 500, 1000, 1500]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for n_f in n_fL:
            #print("for features: ",n_f)
            DTZeroOneLossList=[]
            BTZeroOneLossList=[]
            RFZeroOneLossList=[]
            BSTZeroOneLossList=[]
            SVMZeroOneLossList=[]
            
            for trial in range(10):
                #print("for trial: ",trial)
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                n_features=n_f
                featureListNew=construct_features(dictTrainFile,n_features)
                vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
                vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
                
                train_data=vectorTupleTrain
                test_data=vectorTupleTest
                max_depth=10
                min_size=10
                
                #learn DT
                #print("learning DT")
                misclassify=0
                totalClassify=len(test_data)
                tree_node = build_tree(train_data, max_depth, min_size,False,n_features,featureListNew,False)
                for key in test_data:
                    prediction = predict(tree_node, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("Dt loss: ",zeroOneLoss)
                DTZeroOneLossList.append(zeroOneLoss)
                #end DT
                
                #learn BT
                #print("learning BT")
                number_trees=50
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,False,n_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BT loss: ",zeroOneLoss)
                BTZeroOneLossList.append(zeroOneLoss)
                #end BT
                
                #learn RF
                #print("learning RF")
                number_trees=50
                no_of_features=int(sqrt(len(featureListNew)))
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,True,no_of_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("RF loss: ",zeroOneLoss)
                RFZeroOneLossList.append(zeroOneLoss)
                #end RF
                
                #learn BST
                #print("learning BST")
                min_size=min_size/len(train_data)
                number_trees=50
        
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for key in train_data:
                    train_data[key]=train_data[key]+(1/len(train_data),)
                
                for i in range(number_trees):
                    weighted_error=0
                    sum_new_weight=0
                    tree_node=build_tree(train_data,max_depth, min_size,False,n_features,featureListNew,True)
                    
                    for key in train_data:
                        prediction = predict(tree_node, train_data[key])
                        if(prediction==0):
                            prediction=-1
                        if(train_data[key][0]=='0'):
                            actual=-1
                        else:
                            actual=1
                        if(prediction!=actual):             #misclassified training event
                            weighted_error=weighted_error+train_data[key][2]
                        if(len(train_data[key])<5):
                            train_data[key]=train_data[key]+(prediction,actual)
                        else:
                            lst=[]
                            lst=list(train_data[key])
                            lst[3]=prediction
                            lst[4]=actual
                            train_data[key]=tuple(lst)
                    #print("round: ",i,"train_data: ",train_data)
                    #generated err
                    if(weighted_error==0):
                        log=math.log((1+sys.float_info.epsilon)/(sys.float_info.epsilon))
                    else:
                        log=math.log((1-weighted_error)/weighted_error)
                    alpha=(0.5)*log
                    tree_node['alpha']=alpha
                    #print("round: ",i," :alpha: ",alpha)
                    tree_node_list.append(tree_node)
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=(train_data[key][2])*(math.exp(-(alpha*train_data[key][3]*train_data[key][4])))
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=(train_data[key][2])*(alpha*train_data[key][3]*train_data[key][4])
                        sum_new_weight=sum_new_weight+train_data[key][2]
                    #renormalize again
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=train_data[key][2]/sum_new_weight
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=train_data[key][2]/sum_new_weight
                    #print("round: ",i,"train_data: ",train_data)
                #complete all trees
                for key in test_data:
                    prediction = boosting_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BST loss: ",zeroOneLoss)
                BSTZeroOneLossList.append(zeroOneLoss)
                #end BST
                
                #start SVM
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                #print("SVM loss: ",zeroOneLoss)
                SVMZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            DTavgZeroOneLoss=numpy.average(DTZeroOneLossList)
            DTstdZeroOneLoss=numpy.std(DTZeroOneLossList)/math.sqrt(10)
            DTavgZeroOneLossList.append(DTavgZeroOneLoss)
            DTstdZeroOneLossList.append(DTstdZeroOneLoss)
            BTavgZeroOneLoss=numpy.average(BTZeroOneLossList)
            BTstdZeroOneLoss=numpy.std(BTZeroOneLossList)/math.sqrt(10)
            BTavgZeroOneLossList.append(BTavgZeroOneLoss)
            BTstdZeroOneLossList.append(BTstdZeroOneLoss)
            RFavgZeroOneLoss=numpy.average(RFZeroOneLossList)
            RFstdZeroOneLoss=numpy.std(RFZeroOneLossList)/math.sqrt(10)
            RFavgZeroOneLossList.append(RFavgZeroOneLoss)
            RFstdZeroOneLossList.append(RFstdZeroOneLoss)
            BSTavgZeroOneLoss=numpy.average(BSTZeroOneLossList)
            BSTstdZeroOneLoss=numpy.std(BSTZeroOneLossList)/math.sqrt(10)
            BSTavgZeroOneLossList.append(BSTavgZeroOneLoss)
            BSTstdZeroOneLossList.append(BSTstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            
        #after n_fL list
        print("DTavgZeroOneLossList : ",DTavgZeroOneLossList)
        print("DTstdZeroOneLossList : ",DTstdZeroOneLossList)
        print("BTavgZeroOneLossList : ",BTavgZeroOneLossList)
        print("BTstdZeroOneLossList : ",BTstdZeroOneLossList)
        print("RFavgZeroOneLossList : ",RFavgZeroOneLossList)
        print("RFstdZeroOneLossList : ",RFstdZeroOneLossList)
        print("BSTavgZeroOneLossList : ",BSTavgZeroOneLossList)
        print("BSTstdZeroOneLossList : ",BSTstdZeroOneLossList)
        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
        
        grp.figure(2)
        grp.errorbar(n_fL, DTavgZeroOneLossList, DTstdZeroOneLossList,  marker='^',  label = "DT 0-1 loss")
        grp.errorbar(n_fL, BTavgZeroOneLossList, BTstdZeroOneLossList,  marker='^',  label = "BT 0-1 loss")
        grp.errorbar(n_fL, RFavgZeroOneLossList, RFstdZeroOneLossList,  marker='^',  label = "RF 0-1 loss")
        grp.errorbar(n_fL, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.errorbar(n_fL, BSTavgZeroOneLossList, BSTstdZeroOneLossList,  marker='^',  label = "BST 0-1 loss")
        grp.xlabel('Feature size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('feature_size_loss_q2.png')
    
    elif(ques==3):
        ##INCREMENTAL partition
        ##different depth
        #print("start analysis 3")
        DTavgZeroOneLossList=[]
        DTstdZeroOneLossList=[]
        BTavgZeroOneLossList=[]
        BTstdZeroOneLossList=[]
        RFavgZeroOneLossList=[]
        RFstdZeroOneLossList=[]
        BSTavgZeroOneLossList=[]
        BSTstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        
        perc=0.25
        depthL=[5, 10, 15, 20]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for depth_new in depthL:
            #print("for depth: ",depth_new)
            DTZeroOneLossList=[]
            BTZeroOneLossList=[]
            RFZeroOneLossList=[]
            BSTZeroOneLossList=[]
            SVMZeroOneLossList=[]
            
            for trial in range(10):
                #print("trial no: ",trial)
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                n_features=1000
                featureListNew=construct_features(dictTrainFile,n_features)
                vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
                vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
                
                train_data=vectorTupleTrain
                test_data=vectorTupleTest
                max_depth=depth_new
                min_size=10
                
                #learn DT
                #print("learning DT")
                misclassify=0
                totalClassify=len(test_data)
                tree_node = build_tree(train_data, max_depth, min_size,False,n_features,featureListNew,False)
                for key in test_data:
                    prediction = predict(tree_node, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("Dt loss: ",zeroOneLoss)
                DTZeroOneLossList.append(zeroOneLoss)
                #end DT
                
                #learn BT
                #print("learning BT")
                number_trees=50
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,False,n_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BT loss: ",zeroOneLoss)
                BTZeroOneLossList.append(zeroOneLoss)
                #end BT
                
                #learn RF
                #print("learning RF")
                number_trees=50
                no_of_features=int(sqrt(len(featureListNew)))
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,True,no_of_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("RF loss: ",zeroOneLoss)
                RFZeroOneLossList.append(zeroOneLoss)
                #end RF
                
                #learn BST
                #print("learning BST")
                min_size=min_size/len(train_data)
                number_trees=50
        
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for key in train_data:
                    train_data[key]=train_data[key]+(1/len(train_data),)
                
                for i in range(number_trees):
                    weighted_error=0
                    sum_new_weight=0
                    tree_node=build_tree(train_data,max_depth, min_size,False,n_features,featureListNew,True)
                    
                    for key in train_data:
                        prediction = predict(tree_node, train_data[key])
                        if(prediction==0):
                            prediction=-1
                        if(train_data[key][0]=='0'):
                            actual=-1
                        else:
                            actual=1
                        if(prediction!=actual):             #misclassified training event
                            weighted_error=weighted_error+train_data[key][2]
                        if(len(train_data[key])<5):
                            train_data[key]=train_data[key]+(prediction,actual)
                        else:
                            lst=[]
                            lst=list(train_data[key])
                            lst[3]=prediction
                            lst[4]=actual
                            train_data[key]=tuple(lst)
                    #print("round: ",i,"train_data: ",train_data)
                    #generated err
                    if(weighted_error==0):
                        log=math.log((1+sys.float_info.epsilon)/(sys.float_info.epsilon))
                    else:
                        log=math.log((1-weighted_error)/weighted_error)
                    alpha=(0.5)*log
                    tree_node['alpha']=alpha
                    #print("round: ",i," :alpha: ",alpha)
                    tree_node_list.append(tree_node)
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=(train_data[key][2])*(math.exp(-(alpha*train_data[key][3]*train_data[key][4])))
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=(train_data[key][2])*(alpha*train_data[key][3]*train_data[key][4])
                        sum_new_weight=sum_new_weight+train_data[key][2]
                    #renormalize again
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=train_data[key][2]/sum_new_weight
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=train_data[key][2]/sum_new_weight
                    #print("round: ",i,"train_data: ",train_data)
                #complete all trees
                for key in test_data:
                    prediction = boosting_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BST loss: ",zeroOneLoss)
                BSTZeroOneLossList.append(zeroOneLoss)
                #end BST
                
                #start SVM
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                #print("SVM loss: ",zeroOneLoss)
                SVMZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            DTavgZeroOneLoss=numpy.average(DTZeroOneLossList)
            DTstdZeroOneLoss=numpy.std(DTZeroOneLossList)/math.sqrt(10)
            DTavgZeroOneLossList.append(DTavgZeroOneLoss)
            DTstdZeroOneLossList.append(DTstdZeroOneLoss)
            BTavgZeroOneLoss=numpy.average(BTZeroOneLossList)
            BTstdZeroOneLoss=numpy.std(BTZeroOneLossList)/math.sqrt(10)
            BTavgZeroOneLossList.append(BTavgZeroOneLoss)
            BTstdZeroOneLossList.append(BTstdZeroOneLoss)
            RFavgZeroOneLoss=numpy.average(RFZeroOneLossList)
            RFstdZeroOneLoss=numpy.std(RFZeroOneLossList)/math.sqrt(10)
            RFavgZeroOneLossList.append(RFavgZeroOneLoss)
            RFstdZeroOneLossList.append(RFstdZeroOneLoss)
            BSTavgZeroOneLoss=numpy.average(BSTZeroOneLossList)
            BSTstdZeroOneLoss=numpy.std(BSTZeroOneLossList)/math.sqrt(10)
            BSTavgZeroOneLossList.append(BSTavgZeroOneLoss)
            BSTstdZeroOneLossList.append(BSTstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            
        #after depthL list
        print("DTavgZeroOneLossList : ",DTavgZeroOneLossList)
        print("DTstdZeroOneLossList : ",DTstdZeroOneLossList)
        print("BTavgZeroOneLossList : ",BTavgZeroOneLossList)
        print("BTstdZeroOneLossList : ",BTstdZeroOneLossList)
        print("RFavgZeroOneLossList : ",RFavgZeroOneLossList)
        print("RFstdZeroOneLossList : ",RFstdZeroOneLossList)
        print("BSTavgZeroOneLossList : ",BSTavgZeroOneLossList)
        print("BSTstdZeroOneLossList : ",BSTstdZeroOneLossList)
        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
        
        grp.figure(3)
        grp.errorbar(depthL, DTavgZeroOneLossList, DTstdZeroOneLossList,  marker='^',  label = "DT 0-1 loss")
        grp.errorbar(depthL, BTavgZeroOneLossList, BTstdZeroOneLossList,  marker='^',  label = "BT 0-1 loss")
        grp.errorbar(depthL, RFavgZeroOneLossList, RFstdZeroOneLossList,  marker='^',  label = "RF 0-1 loss")
        grp.errorbar(depthL, BSTavgZeroOneLossList, BSTstdZeroOneLossList,  marker='^',  label = "BST 0-1 loss")
        grp.errorbar(depthL, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.xlabel('Depth size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('depth_size_loss_q3.png')
        
    elif(ques==4):
        ##INCREMENTAL partition
        ##different trees
        #print("start analysis 4")
        DTavgZeroOneLossList=[]
        DTstdZeroOneLossList=[]
        BTavgZeroOneLossList=[]
        BTstdZeroOneLossList=[]
        RFavgZeroOneLossList=[]
        RFstdZeroOneLossList=[]
        BSTavgZeroOneLossList=[]
        BSTstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        
        perc=0.25
        treeL=[10, 25, 50, 100]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for tree_new in treeL:
            #print("for tree size: ",tree_new)
            DTZeroOneLossList=[]
            BTZeroOneLossList=[]
            RFZeroOneLossList=[]
            BSTZeroOneLossList=[]
            SVMZeroOneLossList=[]
            
            for trial in range(10):
                #print("trial no: ",trial)
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                n_features=1000
                featureListNew=construct_features(dictTrainFile,n_features)
                vectorTupleTrain=construct_train_f_vectors(dictTrainFile,featureListNew)
                vectorTupleTest=construct_test_f_vectors(dictTestFile,featureListNew)
                
                train_data=vectorTupleTrain
                test_data=vectorTupleTest
                max_depth=10
                min_size=10
                
                #learn DT
                #print("learning DT")
                misclassify=0
                totalClassify=len(test_data)
                tree_node = build_tree(train_data, max_depth, min_size,False,n_features,featureListNew,False)
                for key in test_data:
                    prediction = predict(tree_node, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("Dt loss: ",zeroOneLoss)
                DTZeroOneLossList.append(zeroOneLoss)
                #end DT
                
                #learn BT
                #print("learning BT")
                number_trees=tree_new
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,False,n_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BT loss: ",zeroOneLoss)
                BTZeroOneLossList.append(zeroOneLoss)
                #end BT
                
                #learn RF
                #print("learning RF")
                number_trees=tree_new
                no_of_features=int(sqrt(len(featureListNew)))
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for i in range(number_trees):
                    sample_dataset=sample_rep(vectorTupleTrain)
                    tree_node=build_tree(sample_dataset,max_depth, min_size,True,no_of_features,featureListNew,False)
                    tree_node_list.append(tree_node)
                for key in test_data:
                    prediction = bagging_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("RF loss: ",zeroOneLoss)
                RFZeroOneLossList.append(zeroOneLoss)
                #end RF
                
                #learn BST
                #print("learning BST")
                min_size=min_size/len(train_data)
                number_trees=tree_new
        
                tree_node_list=[]
                misclassify=0
                totalClassify=len(test_data)
                for key in train_data:
                    train_data[key]=train_data[key]+(1/len(train_data),)
                
                for i in range(number_trees):
                    weighted_error=0
                    sum_new_weight=0
                    tree_node=build_tree(train_data,max_depth, min_size,False,n_features,featureListNew,True)
                    
                    for key in train_data:
                        prediction = predict(tree_node, train_data[key])
                        if(prediction==0):
                            prediction=-1
                        if(train_data[key][0]=='0'):
                            actual=-1
                        else:
                            actual=1
                        if(prediction!=actual):             #misclassified training event
                            weighted_error=weighted_error+train_data[key][2]
                        if(len(train_data[key])<5):
                            train_data[key]=train_data[key]+(prediction,actual)
                        else:
                            lst=[]
                            lst=list(train_data[key])
                            lst[3]=prediction
                            lst[4]=actual
                            train_data[key]=tuple(lst)
                    #print("round: ",i,"train_data: ",train_data)
                    #generated err
                    if(weighted_error==0):
                        log=math.log((1+sys.float_info.epsilon)/(sys.float_info.epsilon))
                    else:
                        log=math.log((1-weighted_error)/weighted_error)
                    alpha=(0.5)*log
                    tree_node['alpha']=alpha
                    #print("round: ",i," :alpha: ",alpha)
                    tree_node_list.append(tree_node)
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=(train_data[key][2])*(math.exp(-(alpha*train_data[key][3]*train_data[key][4])))
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=(train_data[key][2])*(alpha*train_data[key][3]*train_data[key][4])
                        sum_new_weight=sum_new_weight+train_data[key][2]
                    #renormalize again
                    for key in train_data:
                        lst=[]
                        lst=list(train_data[key])
                        lst[2]=train_data[key][2]/sum_new_weight
                        train_data[key]=tuple(lst)
                        #train_data[key][2]=train_data[key][2]/sum_new_weight
                    #print("round: ",i,"train_data: ",train_data)
                #complete all trees
                for key in test_data:
                    prediction = boosting_predict(tree_node_list, test_data[key])
                    if(prediction!=int(test_data[key][0])):
                        misclassify= misclassify +1
                zeroOneLoss=misclassify/totalClassify
                #print("BST loss: ",zeroOneLoss)
                BSTZeroOneLossList.append(zeroOneLoss)
                #end BST
                
                #start SVM
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                #print("SVM loss: ",zeroOneLoss)
                SVMZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            DTavgZeroOneLoss=numpy.average(DTZeroOneLossList)
            DTstdZeroOneLoss=numpy.std(DTZeroOneLossList)/math.sqrt(10)
            DTavgZeroOneLossList.append(DTavgZeroOneLoss)
            DTstdZeroOneLossList.append(DTstdZeroOneLoss)
            BTavgZeroOneLoss=numpy.average(BTZeroOneLossList)
            BTstdZeroOneLoss=numpy.std(BTZeroOneLossList)/math.sqrt(10)
            BTavgZeroOneLossList.append(BTavgZeroOneLoss)
            BTstdZeroOneLossList.append(BTstdZeroOneLoss)
            RFavgZeroOneLoss=numpy.average(RFZeroOneLossList)
            RFstdZeroOneLoss=numpy.std(RFZeroOneLossList)/math.sqrt(10)
            RFavgZeroOneLossList.append(RFavgZeroOneLoss)
            RFstdZeroOneLossList.append(RFstdZeroOneLoss)
            BSTavgZeroOneLoss=numpy.average(BSTZeroOneLossList)
            BSTstdZeroOneLoss=numpy.std(BSTZeroOneLossList)/math.sqrt(10)
            BSTavgZeroOneLossList.append(BSTavgZeroOneLoss)
            BSTstdZeroOneLossList.append(BSTstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            
        #after depthL list
        print("DTavgZeroOneLossList : ",DTavgZeroOneLossList)
        print("DTstdZeroOneLossList : ",DTstdZeroOneLossList)
        print("BTavgZeroOneLossList : ",BTavgZeroOneLossList)
        print("BTstdZeroOneLossList : ",BTstdZeroOneLossList)
        print("RFavgZeroOneLossList : ",RFavgZeroOneLossList)
        print("RFstdZeroOneLossList : ",RFstdZeroOneLossList)
        print("BSTavgZeroOneLossList : ",BSTavgZeroOneLossList)
        print("BSTstdZeroOneLossList : ",BSTstdZeroOneLossList)
        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
        
        grp.figure(4)
        grp.errorbar(treeL, DTavgZeroOneLossList, DTstdZeroOneLossList,  marker='^',  label = "DT 0-1 loss")
        grp.errorbar(treeL, BTavgZeroOneLossList, BTstdZeroOneLossList,  marker='^',  label = "BT 0-1 loss")
        grp.errorbar(treeL, RFavgZeroOneLossList, RFstdZeroOneLossList,  marker='^',  label = "RF 0-1 loss")
        grp.errorbar(treeL, BSTavgZeroOneLossList, BSTstdZeroOneLossList,  marker='^',  label = "BST 0-1 loss")
        grp.errorbar(treeL, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.xlabel('Number of trees')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('tree_size_loss_q4.png')
        
    
else:
    print("Number of arguments is not equal to four. Hence invalid input!!")
    exit()
    
                            
