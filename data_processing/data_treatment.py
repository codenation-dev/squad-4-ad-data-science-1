import pandas as pd
import numpy as np

columns_with_duplicated_info = ["idade_emp_cat", "de_faixa_faturamento_estimado", "de_faixa_faturamento_estimado_grupo"]
columns_with_treated_info = ["dt_situacao"]

def drop_low_significancy_columns(df, threshold = 1):
    """Drops columns with low significancy - a small number of unique values
    
    Arguments:
        df {DataFrame} -- Input DataFrame
    
    Keyword Arguments:
        threshold {int} -- Threshold for dropping columns with equal or less than (default: {1})
    
    Returns:
        DataFrame -- DataFrame with dropped columns
    """
    total_shape =
    return df.drop(columns = df.columns[df.nunique() <= threshold])

def drop_columns(df, col_list = columns_with_duplicated_info):
    """Drop List of Given Columns
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
    
    Keyword Arguments:
        col_list {str list} -- List of columns names with all columns to be dropped (default: {columns_with_duplicated_info})
    
    Returns:
        DataFrame -- Returns the same Dataframe without those rows
    """
    return df.drop(columns = columns_with_duplicated_info)


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


def fill_with_mode(df, col):
    """Fills with mode of the given column
    
    Arguments:
        df {DataFrame} -- Pandas DataFrame
        col {str} -- name of the column
    
    Returns:
        DataFrame -- DataFrame with given column filled with its mode
    """
    df.loc[:,col].fillna(df[col].mode()[0], inplace = True)
    return df

def treat_dt(df, col):
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
    pass



def main_pipeline(df):
    """Runs extensively all dataframe formatting pipeline
    
    Arguments:
        df {DataFrame} -- Full RAW Market Dataframe

    Returns:
        DataFrame -- Full DataFrame with features treated, nulls formatted and so and on
    """

    df = drop_columns(df)
    df = fill_faturamento_estimado(df)
    df = fill_with_zero(df, "(funcionarios|veiculo|idade)", True)
    df = fill_with_mode(df, "dt_situacao")
    df = treat_dt(df, "dt_situacao")
    df = drop_low_significancy_columns(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("../workspace/data/estaticos_market.csv", index_col = 0)

    df = main_pipeline(df)
    print(df.head())
    print((df.isnull().sum()>0).sum())