import fire
from recommender_system_leads import config  # noqa


def features(**kwargs):
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")
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
    # Import Data treatment function:
    import data_treatment
    # Run Treatment and load data:
    df = data_treatment.fetch_market(2)
    # Generate Dataframes:
    # Market:
    market = df.copy()
    # Portfolio
    portfolio = market[market['cliente_flag'] == 1]
    # Market without portfolio
    market_without_clients = market[market['cliente_flag'] == 0]
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
    # Kfold to compare models:
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=3, random_state=random_vl)

def train(**kwargs):
    print("==> TRAINING YOUR MODEL!")
    # Inputs for models:
    X_data = dataset.drop(['cliente_flag'], axis = 1)
    Y_target = dataset['cliente_flag']
    # Model:
    dtree = DecisionTreeClassifier(random_state = random_vl)
    # Best model:
    best_model = dtree
    # Fit train data:
    best_model.fit(X_data, Y_target)

def metadata(**kwargs):
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")
    # confusion matrix:
    print (confusion_matrix(market['cliente_flag'],pred))
    # Classification report:
    print (classification_report(market['cliente_flag'],pred))


def predict(input_data):
    # prediction:
    pred = best_model.predict(market_to_predict)
    # Count predictions:
    predicted_market = market.copy()
    predicted_market['pred'] = pred
    # Get non-customers predict as customer two or more times:
    recommendation = predicted_market.copy()
    recommendation = recommendation[recommendation['pred'] == 1]
    # Delete current customers:
    recommendation['Tr_exc'] = recommendation['id'].isin(portfolio['id'])
    recommendation = recommendation[recommendation['Tr_exc'] != True]
    recommendation = recommendation.drop(['Tr_exc'], axis = 1)
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
    rec_sys_output = recommendation[['id','prob']]
    print("==> PREDICT DATASET {}".format(input_data))


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running recommender_system_leads by ")
    features(**kwargs)  # generate dataset for training
    train(**kwargs)     # training model and save to filesystem
    metadata(**kwargs)  # performance report


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
