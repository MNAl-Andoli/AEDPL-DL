from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import Writer


#path='results/MNIST/Outlier_Detection_FGSM.txt'
path='C:/Users/02729/OneDrive - Universiti Teknikal Malaysia Melaka\Desktop/IEEE Access paper/SVHN/Outlier_Detection_FGSM.txt'

def Random_Forest(X_train, X_test, y_train, y_test, adv_reg_images=""):
    from sklearn.ensemble import RandomForestClassifier

    # Create a Random Forest classifier
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf_rf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf_rf.predict(X_test)

    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("RF detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="RF:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)
    
    #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        y_pred = clf_rf.predict(adv_reg_images)
        
    return y_pred

def KNN(X_train, X_test, y_train, y_test, adv_reg_images=""):
    from sklearn.neighbors import KNeighborsClassifier

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier on the training data
    knn.fit(X_train, y_train)

    # Predict the classes of the testing data
    y_pred = knn.predict(X_test)

    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("KNN detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="KNN:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)
    
     #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        y_pred = knn.predict(adv_reg_images)
        
    return y_pred   



def DT(X_train, X_test, y_train, y_test, adv_reg_images=""):
    
    from sklearn.tree import DecisionTreeClassifier

    # Define the Decision Tree classifier
    clf_dt = DecisionTreeClassifier()

    # Train the model
    clf_dt.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = clf_dt.predict(X_test)
    
    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("DT detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="DT:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)
    
    #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        y_pred = clf_dt.predict(adv_reg_images)
        
    return y_pred
    


def XGB(X_train, X_test, y_train, y_test, adv_reg_images=""):
    import xgboost as xgb

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Define the parameters for XGBoost
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'multi:softmax',
        'num_class': 3
    }

    # Train the model
    xgb_model = xgb.train(params, dtrain, num_boost_round=10)

    # Make predictions on the test set
    y_pred = xgb_model.predict(dtest)

    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("XGB detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="XGB:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)
    
    #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        y_pred = xgb_model.predict(xgb.DMatrix(adv_reg_images))
        
    return y_pred




def GBM(X_train, X_test, y_train, y_test, adv_reg_images=""):
    from sklearn.ensemble import GradientBoostingClassifier
    # Create a GBM classifier instance
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Train the GBM classifier on the training set
    gbm.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = gbm.predict(X_test)

    # Calculate the accuracy of the GBM classifier on the testing set
    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("GBM detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="GBM:\n acc, prc, rec, f1\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) +"\n" 
    Writer.write_results(results, path)
    
    #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        y_pred = gbm.predict(adv_reg_images)
        
    return y_pred



    
    
def NN(X_train, X_test, y_train, y_test,eps_FSGM, attack_type, adv_reg_images=""):

    from tensorflow.keras import models, layers

    # Define the neural network architecture
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50)


    #predict the adversarial images
    adv_images_pred = model.predict(X_test)

    # Convert predicted probabilities to binary labels (0 or 1)
    adv_images_pred_binary = np.round(adv_images_pred)
    y_pred=adv_images_pred_binary

    # Calculate the accuracy of the NN classifier on the testing set
    # Evaluate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    prc=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print("NN detection: acc, prc, rec, f1:", acc, prc, rec, f1)
    
    #write the results
    results ="====================\n acc, prc, rec, f1, eps_FSGM, attack_type,\n"
    results +=str(acc) + "," + str(prc) + "," + str(rec)+ "," + str(f1) + ',' + str(eps_FSGM) + "," + attack_type +"\n" 
    Writer.write_results(results, path)
    
    #to makke prediction for adversarial images
    if(adv_reg_images!=""):
        #predict the adversarial images
        adv_images_pred = model.predict(adv_reg_images)

        # Convert predicted probabilities to binary labels (0 or 1)
        adv_images_pred_binary = np.round(adv_images_pred)
        
    return adv_images_pred_binary

