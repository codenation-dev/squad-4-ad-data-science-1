

columns_with_duplicated_info = ["idade_emp_cat", "de_faixa_faturamento_estimado", "de_faixa_faturamento_estimado_grupo"]

def drop_low_significancy_columns(df, threshold = 1):
    """Drops columns with low significancy - a small number of unique values
    
    Arguments:
        df {DataFrame} -- Input DataFrame
    
    Keyword Arguments:
        threshold {int} -- Threshold for dropping columns with equal or less than (default: {1})
    
    Returns:
        DataFrame -- DataFrame with dropped columns
    """
    return df.drop(columns = df.columns[df.nunique() <= threshold])

def drop_duplicated_columns(df, col_list = columns_with_duplicated_info):
    return df.drop(columns = columns_with_duplicated_info)


def fill_faturamento_estimado(df):
    df.loc[df["sum_faturamento_estimado_coligadas"].isnull(), ["teste_faturamento"] ] = df.vl_faturamento_estimado_aux + df.vl_faturamento_estimado_grupo_aux
    return df

