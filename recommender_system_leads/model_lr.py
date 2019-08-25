from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import fire
import pandas as pd
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/marcelo/Documents/codenation_squad4/squad-4-ad-data-science-1/data_processing/')
import data_treatment as dt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


##Get train and test dataset for pradictions
def get_train_test(dataset):
    portfolio = portfolio1[portfolio1['cliente_flag'] == 1]
    market = portfolio1[portfolio1['cliente_flag'] == 0]

    market_sample = market.sample(portfolio.shape[0])

    portfolio_num = portfolio._get_numeric_data()
    market_num =  market._get_numeric_data()

    portfolio_train = pd.concat([portfolio_num, market_num], sort = False)

    portfolio_train = portfolio_train.fillna(0)
    market_test = market.fillna(0)

    return(portfolio_train, market_test)


def get_importance(dataset):
    logreg = LogisticRegression()

    rfe = RFE(logreg, 20)
    rfe = rfe.fit(dataset.drop('cliente_flag', axis=1), dataset['cliente_flag'])

    rfe_support = rfe.support_
    print(rfe.ranking_)

    feat_importance = []
    col_list = dataset.drop('cliente_flag', axis=1).columns.tolist()

    for i in range(0, len(col_list)):
        if(rfe_support[i]== True):
            feat_importance.append(col_list[i])

    return(feat_importance)

def train(**kwargs):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING YOUR MODEL!")


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    NOTE
    ----
    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")


def predict(input_data):
    """Predict: load the trained model and score input_data

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """
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
    #cli()
    #features(market_dir="/home/marcelo/Documents/codenation_squad4/squad-4-ad-data-science-1/analysis/estaticos_market.csv",
     #        portfolio_dir= "/home/marcelo/Documents/codenation_squad4/squad-4-ad-data-science-1/analysis/estaticos_portfolio1.csv")
    portfolio1 = dt.fetch_market(1)
    portfolio_train, market_test = get_train_test(portfolio1)
    feat_importance = get_importance(portfolio_train)
    print(portfolio_train.shape)
    print(market_test.shape)
    print(feat_importance)
    print('Oi')
