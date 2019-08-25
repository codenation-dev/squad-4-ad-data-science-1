print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")
# coding: utf-8

# # Final Challenge Code:Nation Data Science

# Data handling:
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# models:
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
# Metrics:
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.spatial import distance
# Adjust for visualization:
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# ## Load data and apply Pipeline
# Import Data treatment function:
import data_treatment
# Run Treatment and load data:
df = data_treatment.fetch_market(2)
# Transform index to column
df.reset_index(level=0, inplace=True)
df.head(2)
# Generate Dataframes:
# Market:
market = df.copy()
# Portfolio
portfolio = market[market['cliente_flag'] == 1]
# Market without portfolio
market_without_clients = market[market['cliente_flag'] == 0]

# ## Train and test data:
# > ### Create dataset for train and test:
# random value
random_vl=42
# Function to generate non-clients sample
def make_surrogate(df,samples):
    mod_df=pd.DataFrame(columns=df.columns)
    #random.seed(1)
    #seed=np.random.randint(1,10000,size=len(df.columns))
    count_seed=0
    for i in df.columns:
        mod_df[i]=np.array(df[i].sample(n=samples, random_state=random_vl,replace=True))
        count_seed += 1
    return mod_df
# Number of clients:
num_clients = len(portfolio)
# generate non-clients sample balance dataset of train/test:
non_clients_sample = make_surrogate(market_without_clients,num_clients)
# Create dataset with clients and non-clients:
frames = [portfolio, non_clients_sample ]
dataset = pd.concat(frames)
# Drop not treated data (original data):
not_treated_data = df.select_dtypes("object").columns
dataset = dataset.drop(not_treated_data, axis = 1)

print("==> TRAINING YOUR MODEL!")
# # Prediction models:
# ## Models:
# > ### Cross validation:
# random value
random_vl=42
# Kfold to compare models:
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=3, random_state=random_vl)
# Inputs for models:
X_data = dataset.drop(['cliente_flag'], axis = 1)
Y_target = dataset['cliente_flag']
# > ### KNN Classifier:
# Model:
knn_cl = KNeighborsClassifier(n_neighbors=1)
# > ### Decision Tree:
# Model:
dtree = DecisionTreeClassifier(random_state = random_vl)
# > ### SVM:
# Model:
svm_svc = SVC(gamma='auto', random_state = random_vl)

# ## Benchmark:
# Comparative DataFrame:
performance = []
accuracy=[]
classifier = ['KNN', 'Decision Tree', 'SVM']
models = [knn_cl, dtree, svm_svc]
models_name = ['knn_cl', 'dtree', 'svm_svc']
for i in models:
    model = i
    cv_result = cross_val_score(model,X_data, Y_target, cv = kfold, scoring = "accuracy")
    performance.append(cv_result.mean())
    accuracy.append(cv_result)

models_dataframe=pd.DataFrame(performance,index = classifier)
models_dataframe.columns = ['Accuracy']
models_dataframe['Model_name'] = models_name
models_dataframe['Model_description'] = models
models_dataframe

# # Best model:
best_model = models_dataframe.Model_description[models_dataframe['Accuracy'] == models_dataframe['Accuracy'].max()].tolist()
best_model = best_model[0]
print(models_dataframe['Model_description'].iloc[1])

print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")
# ## Apply the best model:
# > ### Predict
# Complete market with customers, non-customers and all data treated:
market_to_predict = market.drop(not_treated_data, axis = 1)
market_to_predict = market_to_predict.drop(['cliente_flag'], axis = 1)
# Fit train data:
best_model.fit(X_data, Y_target)
# prediction:
pred = best_model.predict(market_to_predict)
# Count predictions:
predicted_market = market.copy()
predicted_market['pred'] = pred
predicted_market['pred'].value_counts()
# confusion matrix:
print (confusion_matrix(market['cliente_flag'],pred))
# Classification report:
print (classification_report(market['cliente_flag'],pred))

# # Recommendation:
# ## Total of possible customers:
# Get non-customers predict as customer two or more times:
recommendation = predicted_market.copy()
recommendation = recommendation[recommendation['pred'] == 1]
# Delete current customers:
recommendation['Tr_exc'] = recommendation['id'].isin(portfolio['id'])
recommendation = recommendation[recommendation['Tr_exc'] != True]
recommendation = recommendation.drop(['Tr_exc'], axis = 1)
print("From a total of %d companies, we found similarities with their portfolio in %d."%(market.shape[0]-len(portfolio), recommendation.shape[0]))

# ## Ranking by proximity Centroid value:
# Default Customer Creation:
X = market_to_predict
Y = market['cliente_flag']
# Fit Nearest Centroid
nctrd = NearestCentroid()
nctrd.fit(X, Y)
default_customer = list(nctrd.centroids_[1])
# Get distances between centroid value and predictions:
z = recommendation.drop(not_treated_data, axis=1)
z = z.drop(['cliente_flag', 'pred'], axis = 1)
recommendation['dist_centroid'] = z.apply(lambda x: distance.euclidean(x, default_customer), axis=1)
# Order by distance of_centroid:
recommendation = recommendation.sort_values(by='dist_centroid', ascending=True)
# Normalize distance value between 0 and 1
recommendation['dist_centroid_norm'] = (recommendation['dist_centroid'] -0)/(max(recommendation['dist_centroid'] )-0)
# Prob column:
prob = 1 - recommendation['dist_centroid_norm']
recommendation['prob'] = list(prob)

print("==> PREDICT DATASET")
# ## Recommendations Outputs:
# Outputs Dataframes
recommendation_ids = recommendation['id']
top25_recommendations_ids = recommendation_ids.head(25)
rec_sys_output = recommendation[['id','prob']]
path = r'../recommender_system_leads/predicts/'
rec_sys_output.to_csv(path+'rec_port_diego.csv', index=False)
