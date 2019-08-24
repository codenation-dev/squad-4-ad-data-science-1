import pandas as pd
import numpy as np

import mapping_dicts
from encoding_scaling import dummies, numerical_scaler



def drop_high_null_columns(df, threshold = .6):
    """Drops columns with a high number of nulls
    
    Arguments:
        df {DataFrame} -- Input DataFrame
    
    Keyword Arguments:
        threshold {int} -- Threshold for dropping columns null % with less than threshold(default: {1})
    
    Returns:
        DataFrame -- DataFrame with dropped columns
    """
    return df.loc[:, df.isnull().mean() < threshold]

def drop_rows_with_nulls(df, cols):
    """Drops rows with a nulls for given columns
    
    Arguments:
        df {DataFrame} -- Input DataFrame
        cols {list} -- String list for all rows to be dropped
    
    Returns:
        DataFrame -- DataFrame with dropped rows
    """
    return df.dropna(axis = 0, how = "any", subset = cols)
 


def drop_low_significancy_columns(df, threshold = 1):
    """Drops columns with low significancy - a small number of unique values
    
    Arguments:
        df {DataFrame} -- Input DataFrame
    
    Keyword Arguments:
        threshold {int} -- Threshold for dropping columns with nunique equal or less than threshold (default: {1})
    
    Returns:
        DataFrame -- DataFrame with dropped columns
    """
    return df.drop(columns = df.columns[df.nunique() <= threshold])

def drop_columns(df, col_list):
    """Drop List of Given Columns
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
    
    Keyword Arguments:
        col_list {str list} -- List of columns names with all columns to be dropped (default: {columns_with_duplicated_info})
    
    Returns:
        DataFrame -- Returns the same Dataframe without those rows
    """
    return df.drop(columns = col_list)


def fill_faturamento_estimado(df):
    """Fills expected revenue for the companie's group
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
    
    Returns:
        DataFrame -- Returns full DataFrame with sum_faturamento_estimado_coligadas filled
    """
    df.loc[df["sum_faturamento_estimado_coligadas"].isnull(), ["sum_faturamento_estimado_coligadas"] ] = df.vl_faturamento_estimado_aux + df.vl_faturamento_estimado_grupo_aux
    return df

def fill_with_zero(df, pattern, regexp = False):
    """Fills columns given str patter or regex
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
        pattern {str} -- String pattern
    
    Keyword Arguments:
        regexp {bool} -- Pattern is about a string or not (default: {False})
    
    Returns:
        DataFrame -- DataFrame with given/selected columns filled with 0
    """
    col_list_selector = df.columns.str.contains( pattern , regex = regexp)
    df.loc[:,col_list_selector] = df.loc[:,col_list_selector].fillna(value = 0)

    return df




def treat_datetime(df, col):
    """Treats date or datetime columns to machine learning ready vals
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
        col {str} -- Datetime or timestamp column name
    
    Returns:
        DataFrame -- Full DataFrame with year column and date column converted to numerical
    """
    df[col+"_numeric"] = pd.to_numeric(pd.to_datetime(df[col]))/100000000000
    df[col+"_year"] = pd.to_datetime(df[col]).dt.year
    
    return df

def map_dataframe(df, mapping_dict):
    for k, v in mapping_dict.items():
        df[k] = df[k].map(v)
    return df

def fill_by_dict(df, filling_dict):
    for k, v in filling_dict.items():
        method = v["method"]
        if method == "constant":
            df[k].fillna(method = None, value = v["value"], inplace = True)
        elif method == "mode":
            df[k].fillna(method = None, value = df[k].mode()[0], inplace = True)
        elif method == "mean":
            df[k].fillna(df[k].mean(), inplace = True)
        elif method == "median":
            df[k].fillna(df[k].median(), inplace = True)
    return df

def get_census_income_agg(df):
    return df.groupby(["nm_micro_regiao" ,"nm_meso_regiao"])[["empsetorcensitariofaixarendapopulacao"]].mean()

def generate_additional_label(df, col_list):
    for col in col_list:
        df[col+"_label"] = 0
        df.loc[~ df[col].isnull(), col+"_label"] = 1
        df[col+"_label"] = df[col+"_label"].astype("bool")
    return df

def columns_to_boolean(df, col_list):
    for col in col_list:
        df[col] = df[col].astype("bool")
    return df

def main_pipeline(df):
    """Runs extensively all dataframe formatting pipeline
    
    Arguments:
        df {DataFrame} -- Full RAW Market Dataframe

    Returns:
        DataFrame -- Full DataFrame with features treated, nulls formatted and so and on
    """
    
    df = fill_with_zero(df, "(funcionarios|veiculo)", True)
    df = fill_by_dict(df, mapping_dicts.FILL_BINARY_N_CONTINUOUS)
    df = fill_faturamento_estimado(df)
    df = treat_datetime(df, "dt_situacao")
    df = generate_additional_label(df, mapping_dicts.COLUMNS_WITH_ADDITIONAL_LABEL_FOR_NULL)
    df = map_dataframe(df, mapping_dicts.MAP_TO_NUMERICAL_ENCODING)

    #fill empsetorcensitariofaixarendapopulacao by segment
    emp_summary = get_census_income_agg(df)
    df.loc[df.empsetorcensitariofaixarendapopulacao.isnull(), ["empsetorcensitariofaixarendapopulacao"]] = df.loc[df.empsetorcensitariofaixarendapopulacao.isnull()].apply(lambda x: emp_summary.loc[x["nm_micro_regiao"], x["nm_meso_regiao"]][0], axis = 1)

    #drop
    df = drop_columns(df, col_list = mapping_dicts.COLUMNS_WITH_DUPLICATED_INFO)
    df = drop_columns(df, col_list = mapping_dicts.COLUMNS_WITH_TREATED_INFO)
    df = drop_low_significancy_columns(df)
    df = drop_high_null_columns(df)
    df = drop_rows_with_nulls(df, cols = ["nm_segmento"])



    #data format
    df = columns_to_boolean(df, mapping_dicts.COLUMNS_TO_BOOL)
    df = dummies(df, 'first')
    df = numerical_scaler(df, mapping_dicts.NUMERICAL_COLUMNS_TO_SCALE)

    return df

portifolio_id = {1: r"../workspace/data/estaticos_portfolio1.csv",
                 2: r"../workspace/data/estaticos_portfolio2.csv",
                 3: r"../workspace/data/estaticos_portfolio3.csv",}

def fetch_market(id_portifolio):
    df = pd.read_csv("../workspace/data/estaticos_market.csv", index_col = 0)
    df.set_index("id", inplace = True)

    df = main_pipeline(df)

    df["cliente_flag"] = 0
    df_portfolio = pd.read_csv(portifolio_id[id_portifolio])["id"]

    df.loc[df.index.isin(df_portfolio.tolist()), "cliente_flag"] = 1

    return df

if __name__ == "__main__":
    # df = pd.read_csv("../workspace/data/estaticos_market.csv", index_col = 0)
    # df.set_index("id", inplace = True)

    # df = main_pipeline(df)
    # print(df.head())
    # null_cols = (df.isnull().sum()>0)
    # print(df.loc[:,null_cols].columns)
    # print(df.info(memory_usage = "deep"))

    df = fetch_market(1)
    print(df.info())

    df = fetch_market(2)
    print(df.info())

    df = fetch_market(3)
    print(df.info())