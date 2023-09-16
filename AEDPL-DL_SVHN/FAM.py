# Fuzzy ARTMAP
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from python_artmap import ARTMAPFUZZY
from sklearn.model_selection import KFold
import time
import gc
import Writer
from os.path import exists

#path='results/MNIST/Outlier_Detection_FGSM.txt'
path='results/CIRFAR10/Outlier_Detection_SA.txt'


#def train(x_train,y_train, rhoARTa=0.6, rhoARTb=0.9):
def train(x_train,y_train, rhoARTa=0.6, rhoARTb=0.9):
    ArtMap = ARTMAPFUZZY(x_train,y_train, rhoARTa, rhoARTb)
    ArtMap.train()
    
    del x_train, y_train
    gc.collect()

    return ArtMap

def test(x_test, y_test, ArtMap,adv_reg_images=""):
    
    len_y=int(y_test.shape[0])
    y_predict=[ArtMap.test(x_test[i]) for i in range(len_y)]

    # "To  check if there is an error in values, and replace non value with -1 by previous value"
    count=0
    for i in range(len(y_predict)):
       
        if (y_predict[i] == -1):
            y_predict[i]=y_predict[i-1]
            count=count+1
            
    y_predict[0]={'index': 1, 'ArtB': [1.0, 0.0], 'id': '1000'}
        
    print(count, " values were replaced from ", len(y_predict))

    
    y_predict=[y_predict[i]['index'] for i in range(len(y_predict))]
    #y_predict=[y_predict[i]['index'] for i in range(len_y)]

    #Accuracy
    acc=accuracy_score(y_test, y_predict)
    #print(acc, y_test.shape, len(y_predict))
    prc=precision_score(y_test, y_predict)
    #print(prc, y_test.shape, len(y_predict))
    rec=recall_score(y_test, y_predict)
    #print(recc, y_test.shape, len(y_predict))
    f1=f1_score(y_test, y_predict)
    #print(f1, y_test.shape, len(y_predict))
    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("FAM detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="FAM:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)

    
    #to make prediction for adversarial images fed to model

    if(adv_reg_images!=""):
        y_predict=[ArtMap.test(adv_reg_images[i]) for i in range(adv_reg_images.shape[0])]
        print(y_predict[0])
        y_predict[0]={'index': 1, 'ArtB': [1.0, 0.0], 'id': '1000'}

        # "To  check if there is an error in values, and replace non value with -1 by previous value"
        count=0
        for i in range(len(y_predict)):
            
            
            if (y_predict[i] == -1):
                
                y_predict[i]=y_predict[i-1]



        y_predict=[y_predict[i]['index'] for i in range(adv_reg_images.shape[0])]


    return y_predict

