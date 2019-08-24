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
#from recommender_system_leads import config  # noqa

## Drop Features with number of missing data greater than a threshold
def drop_na(**kwargs):
    data = kwargs.get("data")
    threshold = kwargs.get("threshold")

    row, col = data.shape

    data_na = pd.DataFrame(data.isnull().sum(), columns=["NA"])

    na_feat_list = (data_na[data_na['NA'] > ((row * threshold) / 100)])
    na_feat_list = na_feat_list.index.values

    data = data.drop(na_feat_list, axis=1)

    return(data)

## Fill NA with mean, Standardize and Normalize dataset
def apply_pipeline(dataframe):
    df_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                   ("scaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
                                   ("minmaxscaler", MinMaxScaler(copy=True, feature_range=(0, 1)))])

    pipeline_transformation = df_pipeline.fit_transform(dataframe)
    data_transformed = pd.DataFrame(pipeline_transformation, columns= dataframe.columns)

    return(data_transformed)

## Fill NA, Standardize and Normalize dataset
def data_prep(dataframe):
    data_obj = dataframe[dataframe.select_dtypes(include=['O']).columns].fillna("NA")
    data_num = dataframe._get_numeric_data()
    data_num = apply_pipeline(data_num)

    result = pd.concat([data_obj, data_num], axis=1)
    result = result[dataframe.columns]
    return (result)

##One hot encode list of categoric features
def one_hot_encode(dataframe):

    enc = OneHotEncoder()

    de_ramo_one_hot = enc.fit_transform(dataframe['de_ramo'].values.reshape(-1, 1)).toarray()
    col_list = enc.get_feature_names().tolist()
    col_list = [col.replace('x0', 'ramo') for col in col_list]
    de_ramo_df = pd.DataFrame(de_ramo_one_hot, columns=col_list)

    setor_one_hot = enc.fit_transform(dataframe['setor'].values.reshape(-1, 1)).toarray()
    col_list = enc.get_feature_names().tolist()
    col_list = [col.replace('x0', 'setor') for col in col_list]
    setor_df = pd.DataFrame(setor_one_hot, columns= col_list)

    natureza_juridica_macro_one_hot = enc.fit_transform(dataframe['natureza_juridica_macro'].values.reshape(-1, 1)).toarray()
    col_list = enc.get_feature_names().tolist()
    col_list = [col.replace('x0', 'jur_macro') for col in col_list]
    natureza_juridica_macro_df = pd.DataFrame(natureza_juridica_macro_one_hot, columns=col_list)

    de_nivel_atividade_one_hot = enc.fit_transform(dataframe['de_nivel_atividade'].values.reshape(-1, 1)).toarray()
    col_list = enc.get_feature_names().tolist()
    col_list = [col.replace('x0', 'nvl_atividade') for col in col_list]
    de_nivel_atividade_df = pd.DataFrame(de_nivel_atividade_one_hot, columns=col_list)

    de_saude_tributaria_one_hot = enc.fit_transform(dataframe['de_saude_tributaria'].values.reshape(-1, 1)).toarray()
    col_list = enc.get_feature_names().tolist()
    col_list = [col.replace('x0', 'saude_trib') for col in col_list]
    de_saude_tributaria_df = pd.DataFrame(de_saude_tributaria_one_hot, columns=col_list)

    df_transformed = pd.concat([dataframe, de_ramo_df, setor_df, natureza_juridica_macro_df, de_nivel_atividade_df,
         de_saude_tributaria_df], axis=1, sort=False)

    return(df_transformed)


def features(**kwargs):

    market_dir = kwargs.get("market_dir")
    portfolio_dir = kwargs.get("market_dir")

    market = pd.read_csv(market_dir)
    market = drop_na(data=market,threshold=85)

    data = pd.read_csv(portfolio_dir)
    portfolio = market.loc[market['id'].isin(data['id'])]
    market = market.loc[~market['id'].isin(data['id'])]

    portfolio = data_prep(portfolio)
    market = data_prep(market)

    portfolio = one_hot_encode(portfolio)
    market = one_hot_encode(market)

    print(market.columns)
    print(portfolio.columns)


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
    features(market_dir="/home/marcelo/Documents/codenation_squad4/squad-4-ad-data-science-1/analysis/estaticos_market.csv",
             portfolio_dir= "/home/marcelo/Documents/codenation_squad4/squad-4-ad-data-science-1/analysis/estaticos_portfolio1.csv")
    print('Oi')
