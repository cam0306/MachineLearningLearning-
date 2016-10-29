import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
import pandas as pd
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neigbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is less than a value that coresponds to the total number of comparing groups')
    
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)- np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1]for i in sorted(distances)[:k]]
    vote_resaults =  Counter(votes).most_common(1)[0][0]  
    confidence =  Counter(votes).most_common(1)[0][1]  / k
    return vote_resaults,confidence

    
    
    
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999,inplace = True)
del df['id']

fulldata = df.astype(float).values.tolist()

random.shuffle(fulldata)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = fulldata[:- int(test_size * len(fulldata))]# gets data up to 80%
test_data = fulldata[-int(test_size * len(fulldata)):]#gets data up to 20%

[train_set[int(i[-1])].append(i[:-1]) for i in train_data] # append data of the classification to the id of the classification
[test_set[int(i[-1])].append(i[:-1]) for i in test_data]
           
correct = 0
total = 0

classifications = []

for group in test_set:
    for data in test_set[group]:
        vote,confedence = k_nearest_neigbors(train_set,data,5)
        if vote == group:
            correct += 1
            classifications += [data,vote,'*']
        else:
            classifications += [data,vote,'x']

        total +=1

accuracy = correct/total

print("Accuracy:",accuracy)

    

        #resault = k_nearest_neigbors(dataset, new_features, 3)
#
#print("Claccified as " + resault )
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1], s = 130, color = resault,marker = '*')
#plt.show()