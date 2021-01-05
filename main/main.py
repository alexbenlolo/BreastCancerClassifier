'''
Alexander Benlolo, 1731184
Monday, May 6
R. Vincent, instructor
Final Project
'''
from extra_trees import extra_trees
from knnclassifier import knnclassifier
from classifier import data_item
from random import shuffle

fp = open('breast-cancer-wisconsin.data.txt')
dataset = []
for line in fp:
    fields = line.split(',') #split data into a list
    data = [float(x) for x in fields[1:-1]] #add data (second value till second-to-last value) as floats to list
    if fields[-1] == '2\n': 
        fields[-1] = '0' #replace with 0 for simplicity
        label = int(fields[-1])
    elif fields[-1] == '4\n':
        fields[-1] = '1' #replace with 1 for simplicity
        label = int(fields[-1]) #assign label as integer
    dataset.append(data_item(label, data)) #add data to list

def split_training_test(dataset,number_fold):
    ''' given a dataset of data_item it will split them into correct number of fold '''
    shuffle(dataset) 
    length_dataset = len(dataset)
    length_training = int(((number_fold-1)/number_fold)*length_dataset) #determine length of training set with fraction

    training_set = dataset[:length_training] #assign fraction of dataset as training set
    test_set = dataset[length_training:] #assign fraction of dataset as testing set

    return (training_set,test_set)

def calculate_confusion_matrix(predict_label,true_label):
    ''' Create a 2 by 2 confusion matrix [[TN,FN],[FP,TP]]'''
    confusion_matrix =  [[0,0],[0,0]] #initialize confusion matrix
    for p_label,t_label in zip(predict_label,true_label):
        confusion_matrix[p_label][t_label] += 1 #will appropriately increment according matrix space, using predicted and true labels
    return confusion_matrix

def evaluate_fitting(confusion_matrix):
    ''' calculate the fpr and the tpr based on the confusion matrix'''
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]

    tpr = (tp/(tp+fn)) #tpr formula
    fpr = (fp/(fp+tn)) #fpr formula

    return (tpr,fpr)

def k_fold(dataset,number_fold,M,K,Nmin,P):
    ''' Train the algorithm on each fold, then test and give back an average accuracies measure'''

    total_tpr = 0
    total_fpr = 0

    if P == 0:
        print('Using extra trees:')
    elif P == 1:
        print('Using KNN:')
    for i in range(0,number_fold): 
        print("{} Fold:".format(i))
        training_set, test_set = split_training_test(dataset,number_fold)

        # Training
        print("Training...")
        if P == 0:
            clas = extra_trees(M,K,Nmin) #assign using extra_trees classifier
        elif P == 1:
            clas = knnclassifier() #assign using knnclassifier classifier
        clas.train(training_set) #train clasifier with training data
        # Testing
        print("Testing...")
        test_data = [x.data for x in test_set[:]] #create list with test data from test set
        true_label = [x.label for x in test_set[:]] #create list with test labels from test set
        predict_label = [clas.predict(x) for x in test_data] #create list with predicted test labels from test data
            

        
        confusion_matrix = calculate_confusion_matrix(predict_label,true_label)
        tpr,fpr = evaluate_fitting(confusion_matrix)
        
        total_tpr += tpr #increment total tpr by tpr for one fold
        total_fpr += fpr #increment total fpr by fpr for one fold
        print("TPR = {} and FPR = {}.".format(tpr,fpr))
    average_tpr = total_tpr/number_fold #compute avg tpr over number of folds
    average_fpr = total_fpr/number_fold #compute avg fpr over number of folds
    return (average_tpr,average_fpr)

kfold = 5 #number of folds to split dataset
M = 15 #for extra_trees classifier 
K = 10 #for extra_trees classifier
Nmin = 2 #for extra_trees classifier
P = 0 #in order to use trees classifier first
average_tpr,average_fpr = k_fold(dataset,kfold,M,K,Nmin,P)
print("Trees: Average TPR = {} and Average FPR = {}.".format(average_tpr,average_fpr))
P += 1 #in order to use knn classifier second
average_tpr,average_fpr = k_fold(dataset,kfold,M,K,Nmin,P)
print("KNN: Average TPR = {} and Average FPR = {}.".format(average_tpr,average_fpr))

