# # Final Challenge Code:Nation Data Science

# ## Libraries and Files:
# > ### Packages:

# Data handling:
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# models: 
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
# Metrics:
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.spatial import distance

# > ### Data:
market = pd.read_csv("market.csv")
portfolio = pd.read_csv("portfolio.csv")
dictionary = pd.read_csv("dicionario.csv")

# Adjust for viewing:
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', len(market.columns))
pd.set_option('display.max_rows', len(market))

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# ## T1 - Dataset complete Portfolio:
# The portfolio dataset has only the column 'id', we will complete with the other data that are present in the dataset market.

# DataFrame handling  Step 1:
market_T1= market.copy()

# Creation of id_num column to help manipulation:
market_T1['id_num'] = market_T1['Unnamed: 0']

# Delete column 'Unnamed: 0':
market_T1 = market_T1.drop(['Unnamed: 0'], axis = 1)

# Complete portfolio generation:
portfolio_T1 = pd.DataFrame(portfolio['id'])
portfolio_T1 = pd.merge(portfolio_T1, market_T1, on='id', how='left')

# > ### Shapes(Rows x Columns):
print('Market:', market_T1.shape)
print('Portfolio:', portfolio_T1.shape)

# ## T2 - NA's Handling
# > ### NA's per columns:
# NA's percentage check
qt_na_per_col_portfolio_T1 = pd.DataFrame(portfolio_T1.isna().sum().sort_values(ascending=False).reset_index())
qt_na_per_col_portfolio_T1.columns = ['Column', 'qt_NA']
qt_na_per_col_portfolio_T1['qt_not_NA'] = (len(portfolio_T1) - qt_na_per_col_portfolio_T1['qt_NA'])
qt_na_per_col_portfolio_T1['Percentage_of_NA'] = ((qt_na_per_col_portfolio_T1['qt_NA'] * 100) / len(portfolio_T1)).round(2)

# > ### Exclusion High Percent NA Columns:
# DataFrame preparation Step 2:
market_T2 = market_T1.copy()

# NA Limit
percent_NA = 55

# Selection of all columns that have high NA's:
column_NA_exclusion = qt_na_per_col_portfolio_T1['Column'].loc[qt_na_per_col_portfolio_T1['Percentage_of_NA'] >= percent_NA]
# Exclusion of columns:
market_T2 = market_T2.drop(column_NA_exclusion, axis = 1)

# > ### Data Types:
# Types:
types_market_T2 = pd.DataFrame(market_T2.dtypes).reset_index()
types_market_T2.columns = ['Columns', 'types']

# type lists:
type_float = list (market_T2.select_dtypes (include = ['float']). columns)
type_int = list (market_T2.select_dtypes (include = ['int']). columns)
type_bool = list (market_T2.select_dtypes (include = ['bool']). columns)
type_object = list (market_T2.select_dtypes (include = ['object']). columns)

# Removal of identifying columns from type lists:
type_object.remove('id')
type_int.remove('id_num')

# Remove dt_situacao
type_object.remove('dt_situacao')


# > ### Fill NA'S:
# - Numeric variables by 0;
# - Categorical variables by value 'INDETERMINADA';
# Fill Na's
market_T2[type_int] = market_T2[type_int].fillna(0)
market_T2[type_float] = market_T2[type_float].fillna(0)
market_T2[type_object] = market_T2[type_object].fillna('INDETERMINADA')

# Comparative:
print('Starter Market:', market_T1.shape)
print('Market after handling:', market_T2.shape)
print('Unimportant columns deleted:', market_T1.shape[1] - market_T2.shape[1])

# ## T3 - Variable coding:
# DataFrame preparation step 5:
market_T3 = market_T2.copy()
market_T3_a = market_T3[['id', 'id_num']]
# Exclusion of identification variables:
market_T3_b = market_T3.drop(['id', 'id_num'], axis = 1)

# Conversions:
market_T3_b[type_float] = market_T3_b[type_float].astype('int')
market_T3_b[type_bool] = market_T3_b[type_bool].astype('category')
market_T3_b[type_bool] = market_T3_b[type_bool].apply(lambda x: x.cat.codes)
market_T3_b[type_object] = market_T3_b[type_object].astype('category')
market_T3_b[type_object] = market_T3_b[type_object].apply(lambda x: x.cat.codes)

# Market Reconstruction:
market_T3 = pd.DataFrame()
market_T3 = pd.concat([market_T3_a, market_T3_b], axis=1)
market_T3 = market_T3.drop(['dt_situacao'], axis=1)

# Portfolio after T3
portfolio_T3 = pd.DataFrame(portfolio['id'])
portfolio_T3['Tr_exc'] = portfolio_T3['id'].isin(market_T3['id'])
portfolio_T3 = portfolio_T3[portfolio_T3['Tr_exc'] == True]
portfolio_T3 = portfolio_T3.drop(['Tr_exc'], axis = 1)
portfolio_T3 = pd.merge(portfolio_T3, market_T3, on='id', how='left')

# ## T4 - Label clients:

# DataFrame preparation Step 4:
market_T4 = market_T3.copy()
portfolio_T4 = portfolio_T3.copy()

# assign value 1 to customers:
portfolio_T4['cliente'] = 1
# assign value 1 to non-customers:
market_T4['cliente'] = 0
# assign value 1 to portfolio's customers in market:
market_T4.loc[market_T4['id'].isin(portfolio_T4['id']), ['cliente']] = 1

# Data normalization:
min_max_scaler = preprocessing.MinMaxScaler()
market_T4[type_int] = min_max_scaler.fit_transform(market_T4[type_int])
market_T4[type_float] = min_max_scaler.fit_transform(market_T4[type_float])
market_T4[type_object] = min_max_scaler.fit_transform(market_T4[type_object])

# Portfolio after T4
portfolio_T4 = pd.DataFrame(portfolio['id'])
portfolio_T4['Tr_exc'] = portfolio_T4['id'].isin(market_T4['id'])
portfolio_T4 = portfolio_T4[portfolio_T4['Tr_exc'] == True]
portfolio_T4 = portfolio_T4.drop(['Tr_exc'], axis = 1)
portfolio_T4 = pd.merge(portfolio_T4, market_T4, on='id', how='left')

# ## T5 - Train and test data:
# random value
random_vl=42

# > ### Create dataset for train and test:
# DataFrame preparation Step 5:
market_T5 = market_T4.copy()
portfolio_T5 = portfolio_T4.copy()

# market without current customers:
market_T5_without_clients = pd.DataFrame(market_T5['id'])
market_T5_without_clients['Tr_exc'] = market_T5_without_clients['id'].isin(portfolio_T5['id'])
market_T5_without_clients = market_T5_without_clients[market_T5_without_clients['Tr_exc'] != True]
market_T5_without_clients = market_T5_without_clients.drop(['Tr_exc'], axis = 1)
market_T5_without_clients = pd.merge(market_T5_without_clients, market_T5, on='id', how='left')

# Number of customers:
num_clients = len(portfolio_T5)
# get non-customers for balance dataset: 
non_clients_sample = market_T5_without_clients.sample(n=num_clients, random_state = random_vl)

# Create dataset with customers and non-customers
frames = [portfolio_T5, non_clients_sample ]
dataset = pd.concat(frames)

# Market without dataset entries:
market_without_dataset = pd.DataFrame(market_T5['id'])
market_without_dataset['Tr_exc'] = market_without_dataset['id'].isin(dataset['id'])
market_without_dataset = market_without_dataset[market_without_dataset['Tr_exc'] != True]
market_without_dataset = market_without_dataset.drop(['Tr_exc'], axis = 1)
market_without_dataset = pd.merge(market_without_dataset, market_T5, on='id', how='left')

# # Prediction model:
# ## Model:
# Inputs for models:
X_data = dataset.drop(['id', 'id_num','cliente'], axis = 1)
Y_target = dataset['cliente']

# > ### Decision Tree:
# Model:
dtree = DecisionTreeClassifier(random_state = random_vl)

# # Best model:
best_model = dtree

# ## Apply the best model:
# > ### Predict 1

# Fit train data:
best_model.fit(X_data, Y_target)

# Adjust dataset to predict:
predict_market = market_T4.drop(['id', 'id_num','cliente'], axis = 1)
# prediction:
pred = best_model.predict(predict_market)
# Count predictions:
predict_market = market_T4.copy()
predict_market['pred_1'] = pred
predict_market['pred_1'].value_counts()

print('Predict 1')
# confusion matrix:
print (confusion_matrix(market_T4['cliente'],pred))

# Classification report:
print (classification_report(market_T4['cliente'],pred))

# > ### Predict 2
# get new non-customers data
temp_df_pred2 = market_T5_without_clients.copy()
temp_df_pred2['Tr_exc'] = temp_df_pred2['id'].isin(dataset['id'])
temp_df_pred2 = temp_df_pred2[temp_df_pred2['Tr_exc'] != True]
temp_df_pred2 = temp_df_pred2.drop(['Tr_exc'], axis = 1)

# get non-customers for balance dataset: 
non_clients_sample_2 = temp_df_pred2.sample(n=num_clients, random_state = random_vl)

# Create dataset with customers and non-customers
frames_2 = [portfolio_T5, non_clients_sample_2 ]
dataset_2 = pd.concat(frames_2)

# Inputs for models:
X_data = dataset_2.drop(['id', 'id_num','cliente'], axis = 1)
Y_target = dataset_2['cliente']

# Fit train data:
best_model.fit(X_data, Y_target)

# prediction:
predict_market_2 = market_T4.drop(['id', 'id_num','cliente'], axis = 1)
pred = best_model.predict(predict_market_2)
# Count predictions:
predict_market['pred_2'] = pred
predict_market['pred_2'].value_counts()


print('Predict 2')
# confusion matrix:
print (confusion_matrix(market_T4['cliente'],pred))

# Classification report:
print (classification_report(market_T4['cliente'],pred))

# > ### Predict_3
# get new non-customers data
temp_df_pred3 = temp_df_pred2.copy()
temp_df_pred3['Tr_exc'] = temp_df_pred3['id'].isin(dataset_2['id'])
temp_df_pred3 = temp_df_pred3[temp_df_pred3['Tr_exc'] != True]
temp_df_pred3 = temp_df_pred3.drop(['Tr_exc'], axis = 1)

# get non-customers for balance dataset: 
non_clients_sample_3 = temp_df_pred3.sample(n=num_clients, random_state = random_vl)

# Create dataset with customers and non-customers
frames_3 = [portfolio_T5, non_clients_sample_3 ]
dataset_3 = pd.concat(frames_3)

# Inputs for models:
X_data = dataset_3.drop(['id', 'id_num','cliente'], axis = 1)
Y_target = dataset_3['cliente']

# Fit train data:
best_model.fit(X_data, Y_target)

# prediction:
predict_market_3 = market_T4.drop(['id', 'id_num','cliente'], axis = 1)
pred = best_model.predict(predict_market_3)
# Count predictions:
predict_market['pred_3'] = pred
predict_market['pred_3'].value_counts()

print('Predict 3')
# confusion matrix:
print (confusion_matrix(market_T4['cliente'],pred))

# Classification report:
print (classification_report(market_T4['cliente'],pred))

# Feature predict sum
predict_market['predict_sum'] = (predict_market['pred_1'] + predict_market['pred_2'] + predict_market['pred_3'])

# > ### Results of data used in the model: 

# Verify original customers classification:
original_customers = predict_market.loc[predict_market['cliente'] == 1]
((original_customers['pred_1'] + original_customers['pred_2'] + original_customers['pred_3'])).value_counts()

# Verify classification of non-customers in train dataset: 
non_customers_dataset = predict_market.loc[predict_market['cliente'] == 0]

frames_nc = [non_clients_sample, non_clients_sample_2, non_clients_sample_3 ]
dataset_nc = pd.concat(frames_nc)

non_customers_dataset = non_customers_dataset[non_customers_dataset['id'].isin(dataset_nc['id'])]
((non_customers_dataset['pred_1'] + non_customers_dataset['pred_2'] + non_customers_dataset['pred_3'])).value_counts()

# ## Recommendation:

# > ### Total of possible customers:
recommendation = predict_market.copy()
recommendation = recommendation[recommendation['predict_sum'] >= 2]

# Delete current customers:
recommendation['Tr_exc'] = recommendation['id'].isin(portfolio_T5['id'])
recommendation = recommendation[recommendation['Tr_exc'] != True]
recommendation = recommendation.drop(['Tr_exc'], axis = 1)

print("From a total of %d companies, we found similarities with their portfolio in %d."%(market.shape[0]-len(portfolio), recommendation.shape[0]))

# > ### Ranking by proximity Centroid value:

# Default Customer Creation:
X = market_T4.drop(['id','id_num', 'cliente'], axis=1)
Y = market_T4['cliente']
nctrd = NearestCentroid()
nctrd.fit(X, Y)
default_customer = list(nctrd.centroids_[1])

# values default customer:
print(default_customer)

#Get distances between centroid value and predictions:
z = recommendation.drop(['id', 'id_num','cliente', 'pred_1', 'pred_2', 'pred_3','predict_sum' ], axis=1)
recommendation['dist_centroid'] = z.apply(lambda x: distance.euclidean(x, default_customer), axis=1)

# Order by distance of_centroid:
recommendation = recommendation.sort_values(by='dist_centroid', ascending=True)
recommendation['dist_centroid_norm'] = (recommendation['dist_centroid'] -0)/(max(recommendation['dist_centroid'] )-0)

# > ### Recommendation with original values:

# Get id's from recommendation:
recommendation_full = pd.DataFrame()
recommendation_full['id'] = recommendation['id']

# Get id's original values:
recommendation_full = pd.merge(recommendation_full, market_T1, on='id', how='left')

# View all similar possible customers DataFrame

recommendation_full['cliente'] = list(recommendation['cliente'])
recommendation_full['pred_1'] = list(recommendation['pred_1'])
recommendation_full['pred_2'] = list(recommendation['pred_2'])
recommendation_full['pred_3'] = list(recommendation['pred_3'])
recommendation_full['predict_sum'] = list(recommendation['predict_sum'])
recommendation_full['dist_centroid'] = list(recommendation['dist_centroid'])
recommendation_full['dist_centroid_norm'] = list(recommendation['dist_centroid_norm'])

# Recommendation_ids
Recommendation_full_ids = recommendation_full['id']

# > ### Top25 Recommendations:
# Full Top25_recommendations:
Top25_recommendations_ids = recommendation_full['id'].head(25)
Top25_recommendations = recommendation_full.head(25)

# ### Save recommendation in csv files:
# Ids
# Top25_recommendations_ids.to_csv('Top25_recommendations_ids.csv', index=False)
# Recommendation_full_ids.to_csv('Recommendation_full_ids.csv', index=False)

# Completes data
recommendation_full.head(500).to_csv('rec_port1_diego.csv', index=False)
# Top25_recommendations.to_csv('Top25_recommendations.csv', index=False)
print('F I M_')
