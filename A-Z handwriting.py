#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#%% In[1]: PART 1. LOOK AT THE BIG PICTURE

# In[2]: PART 2. GET THE DATA (DONE). LOAD DATA
df = pd.read_csv('../datasets/character.csv')
# In[3]: PART 3. DISCOVER THE DATA TO GAIN INSIGHTS
#region
print('\n____________ Dataset info ____________')
print(df.info())
print('\n____________ Some first data examples ____________')
print(df.head(20)) 
#endregion
# In[4]: PART 4. PREPARE THE DATA 
#region
col = len(df.columns)
print(col)
#%%
#Tách các feature và label ra
X = np.array(df.iloc[:, 1: col])
y = np.array(df.iloc[:,0])
print(X.shape)
#%%
# Scale feature:
# Before scale: feature [0, 255] (0 là màu đen, 255 là màu trắng)
# After scale: feature [0,1] 
X = X.astype('float64')
X = X/255
#%%
# Hàm show image
def show_image(arr):
    arr = np.reshape(arr, (28, 28))
    plt.imshow(arr,cmap='gray')
#%%
#Lấy ra số lượng label
num_of_char = len(np.unique(y))
print(num_of_char)
#%%
# Khởi tạo train và test sets
X_train = np.array([])
y_train = np.array([])
X_test = np.array([])
y_test = np.array([])
# %%
def split_datasets(X, y, num, test_size):
    for i in range(num):
        X_char = X[y==i]            
        y_char = y[y==i]
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_char, y_char, test_size=test_size, random_state=42)
        if (i == 0):
            X_train, y_train, X_test, y_test =  X_train_temp, y_train_temp, X_test_temp, y_test_temp
            continue
        X_train = np.append(X_train, X_train_temp, axis=0)
        X_test = np.append(X_test, X_test_temp, axis=0)
        y_train = np.append(y_train, y_train_temp, axis=0)
        y_test = np.append(y_test, y_test_temp, axis=0)
    
    return (X_train, y_train, X_test, y_test)
#%%
(X_train, y_train, X_test, y_test) = split_datasets(X, y, num_of_char, test_size=0.2)
print('\n____________ Split training and test set ____________')     
print(len(X_train), "training +", len(X_test), "test samples")
# In[5]: PART 5. TRAIN AND EVALUATE MODELS 
#region
# Try KNeighborsClassifier
new_run=True
if new_run:
    #Khởi tạo model
    knn = KNeighborsClassifier(n_neighbors=4, weights="distance")            #accuracy trên tập test 98%
    #Trainning model
    knn.fit(X_train,y_train)
    joblib.dump(knn,'../models/knn_clf')
else:
    knn= joblib.load('../models/knn_clf')
#%%
# Try prediction
sample_id = 253332
print(knn.predict([X_train[sample_id]]))
print('Label: ',y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
# Try RandomForest classifier
new_run=True
if new_run == True:
    #Khởi tạo model
    rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)       #accuracy trên tập test 98%
    #Trainning model
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf,'../models/rf_clf')
else:
    rf_clf = joblib.load('../models/rf_clf')

#%%
#In ra dự đoán
print(rf_clf.predict([X_train[sample_id]]))
#In ra label
print(y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
#Try Softmax Regression
if new_run==True: 
    softmax_reg = LogisticRegression(multi_class='multinomial')
    softmax_reg.fit(X_train,y_train)
    joblib.dump(softmax_reg,'../models/softmax_reg')
else:
    softmax_reg=joblib.load('../models/softmax_reg')
#%%
#In ra dự đoán
print(softmax_reg.predict([X_train[sample_id]]))
#In ra label
print(y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
#Try SGD
#Khởi tạo model
if new_run==True:
    sgd_clf = SGDClassifier(loss='modified_huber',random_state=41)
    sgd_clf.fit(X_train,y_train)
    joblib.dump(sgd_clf,'../models/sgd_clf')
else:
    sgd_clf=joblib.load('../models/sgd_clf')
#%%
#In ra dự đoán
print(sgd_clf.predict([X_train[sample_id]]))
#In ra label
print(y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
# Try Extra-Trees
new_run = True
if new_run:
    extra_trees = ExtraTreesClassifier(n_estimators=20, random_state=41)
    extra_trees.fit(X_train, y_train)
    joblib.dump(extra_trees, '../models/extra_tree_clf')
else:
    extra_trees = joblib.load('../models/extra_tree_clf')
#%%
# In ra dự đoán
print(extra_trees.predict([X_train[sample_id]]))
# In ra label
print(y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
# Try Decision Tree
new_run = True
if new_run:
    decision_tree = DecisionTreeClassifier(random_state=41)
    decision_tree.fit(X_train, y_train)
    joblib.dump(decision_tree, '../models/decision_tree_clf')
else:
    decision_tree = joblib.load('../models/decision_tree_clf')
#%%
# In ra dự đoán
print(decision_tree.predict([X_train[sample_id]]))
# In ra label
print(y_train[sample_id])
print('Label in character format: ',chr(ord('a')+y_train[sample_id]))
show_image(X_train[sample_id])
#%%
# Evaluate on train set
test_train_set = False
if test_train_set:
    rf_train_score = accuracy_score(y_train, rf_clf.predict(X_train))
    knn_train_score = accuracy_score(y_train, knn.predict(X_train))
    softmax_train_score = accuracy_score(y_train, softmax_reg.predict(X_train))
    sgd_train_score = accuracy_score(y_train, sgd_clf.predict(X_train))
    extra_trees_train_score = accuracy_score(y_train, extra_trees.predict(X_train))
    decision_tree_train_score = accuracy_score(y_train, decision_tree.predict(X_train))

    print(f"Accuracy Score in Train Set with Random Forest is: {rf_train_score}")
    print(f"Accuracy Score in Train Set with KNN is: {knn_train_score}")
    print(f"Accuracy Score in Train Set with Softmax Regression is: {softmax_train_score}")
    print(f"Accuracy Score in Train Set with SGD Classification is: {sgd_train_score}")
    print(f"Accuracy Score in Train Set with Extra-Trees is: {extra_trees_train_score}")
    print(f"Accuracy Score in Train Set with Decision Tree is: {decision_tree_train_score}")

#%%
# Evaluate with cross validation
new_run_score=False
if new_run_score:
    knn_accuracies = cross_val_score(knn, X_train, y_train, cv=3, scoring="accuracy")
    rf_accuracies=cross_val_score(rf_clf,X_train,y_train,cv=3,scoring="accuracy")
    softmax_accuracies=cross_val_score(softmax_reg,X_train,y_train,cv=3,scoring="accuracy")
    sgd_accuracies=cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")
    extra_trees_accuracies = cross_val_score(extra_trees,X_train,y_train,cv=3,scoring="accuracy")
    decision_tree_accuracies = cross_val_score(decision_tree,X_train,y_train,cv=3,scoring="accuracy")
    
    joblib.dump(extra_trees_accuracies,'./saved_var/extra_trees_cross_val_score')
    joblib.dump(knn_accuracies,'./saved_var/knn_cross_val_score')
    joblib.dump(rf_accuracies,'./saved_var/rf_cross_val_score')
    joblib.dump(softmax_accuracies,'./saved_var/softmax_cross_val_score')
    joblib.dump(sgd_accuracies,'./saved_var/sgd_cross_val_score')
    joblib.dump(decision_tree_accuracies, './saved_var/decision_tree_cross_val_score')
else:
    knn_accuracies=joblib.load('./saved_var/knn_cross_val_score')
    rf_accuracies=joblib.load('./saved_var/rf_cross_val_score')
    softmax_accuracies=joblib.load('./saved_var/softmax_cross_val_score')
    sgd_accuracies=joblib.load('./saved_var/sgd_cross_val_score')
    extra_trees_accuracies = joblib.load('./saved_var/extra_trees_cross_val_score')
    decision_tree_accuracies = joblib.load('./saved_var/decision_tree_cross_val_score')
print('KNeighborsClassifier cross-validation score: ',knn_accuracies)
print('RandomForestClassifier cross-validation score: ',rf_accuracies)
print('SGDClassifier cross-validation score: ',sgd_accuracies)
print('Extra-Trees cross-validation score: ',extra_trees_accuracies)
print('Decision Tree cross-validation score: ',decision_tree_accuracies)
print('Softmax Regression cross-validation score: ',softmax_accuracies)

#endregion
# In[6]: PART 6. FINE-TUNE MODELS 
#region
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)

#%%
# Random Forest
rf_param_grid = {
    'n_estimators': [30, 35, 45]
}

rf_clf_tune = GridSearchCV(
    estimator=RandomForestClassifier(random_state=41),
    param_grid=rf_param_grid
)


rf_clf_tune.fit(X_train, y_train)
print_search_result(rf_clf_tune, "Random Forest Classifier")
#%%
# Extra-trees
extra_trees_param_grid = {
    'n_estimators': [40, 45, 50]
}

extra_trees_tune = GridSearchCV(
    estimator=ExtraTreesClassifier(random_state=41),
    param_grid=extra_trees_param_grid
)

extra_trees_tune.fit(X_train, y_train)
print_search_result(extra_trees_tune, "Extra-Trees Classifer")
# In[7]: PART 7. ANALYZE AND TEST YOUR SOLUTION
def show_result_model(real_label, predict):
    print(accuracy_score(real_label, predict))
    error_matrix=confusion_matrix(real_label,predict)
    plt.matshow(error_matrix, cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    plt.show()

#%%
# Voting Classifier with KNN, Random Forest and Extra-trees
voting_clf = VotingClassifier(
    estimators=[('knn', KNeighborsClassifier(n_neighbors=4, weights='distance')),
                ('random-forest', RandomForestClassifier(n_estimators=45, random_state=41)),
                ('extra-trees', ExtraTreesClassifier(n_estimators=50))],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
# %%
# Voting Classifier with Random Forest and Extra-trees
voting_clf2 = VotingClassifier(
    estimators=[
        ('random-forest', RandomForestClassifier(n_estimators=45, random_state=41)),
        ('extra-trees', ExtraTreesClassifier(n_estimators=50))
    ],
    voting='soft'
)
voting_clf2.fit(X_train, y_train)
# %%
# Bagging with 4 estimator Random Forest
bagging_clf = BaggingClassifier(
    base_estimator=RandomForestClassifier(n_estimators=45, random_state=41),
    n_estimators=4
)
bagging_clf.fit(X_train, y_train)
# %%
# Stacking with layer 0 is KNN and Extra-trees and final layer is Random Forest
stacking_clf = StackingClassifier(
    estimators=[('knn', KNeighborsClassifier(n_neighbors=4, weights='distance')),
                ('extra-trees', ExtraTreesClassifier(n_estimators=50))],
    final_estimator=RandomForestClassifier(n_estimators=45, random_state=41)
)
stacking_clf.fit(X_train, y_train)
#%% Test Model
# ==================== Base model =================
# KNN
show_result_model(y_test,knn.predict(X_test))
#%%
# Random Forest
show_result_model(y_test, rf_clf.predict(X_test))
#%%
# Softmax Regression
show_result_model(y_test, softmax_reg.predict(X_test))
#%%
# Decision Tree
show_result_model(y_test, decision_tree.predict(X_test))
#%%
# Extra-trees 
show_result_model(y_test, extra_trees.predict(X_test))
#%%
# SGD
show_result_model(y_test, sgd_clf.predict(X_test))
#%%
# ================= Tuned Model =================
#%%
# Random Forest Tuned
show_result_model(y_test, rf_clf_tune.predict(X_test))
#%%
# Extra-Trees Tuned
show_result_model(y_test, extra_trees_tune.predict(X_test))
#%%
# ========== Ensemble Model ============
# %%
# Voting Classifier with KNN, Random Forest and Extra-trees
show_result_model(y_test, voting_clf.predict(X_test))
# %%
# Voting Classifier with Random Forest and Extra-trees
show_result_model(y_test, voting_clf2.predict(X_test))
# %%
# Bagging with 4 estimators of Random Forest
show_result_model(y_test, bagging_clf.predict(X_test))
#%%
# Stacking with layer 0 is KNN and Extra-trees and final layer is Random Forest
show_result_model(y_test, stacking_clf.predict(X_test))

#%%
# ==================== Choose Model ================
# Best Model On Test Set: Stacking Classification
# Code: 
''' 
stacking_clf = StackingClassifier(
    estimators=[('knn', KNeighborsClassifier(n_neighbors=4, weights='distance')),
                ('extra-trees', ExtraTreesClassifier(n_estimators=50))],
    final_estimator=RandomForestClassifier(n_estimators=45, random_state=41)
) 
'''

# Best Model On Test Set an Good about Time Prediction: Voting Classification with Random Forest and Extra-Trees
# Code: 
''' 
voting_clf2 = VotingClassifier(
    estimators=[
        ('random-forest', RandomForestClassifier(n_estimators=45, random_state=41)),
        ('extra-trees', ExtraTreesClassifier(n_estimators=50))
    ],
    voting='soft'
) 
'''
# In[8]: PART 8. LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM






#%%
# ----------------------------------------DONE---------------------------------------
# DATASETS LINK:
# https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format


# REFERENCE LINKS:
# https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
# https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
# %%
