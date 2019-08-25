import numpy as np
import matplotlib
import pandas as pd

# Runs Recommender models Scripts
##import recommender_system_class # CLassification script
##import LR_script_NAME # Logistic Regression script
##import RF_script_NAME # Random Forest script

# Ranking aggregate Function:
def aggregate_rank(quantile_leads = 0.05,
                   out_rank_name = r'../recommender_system_leads/predicts/ag_rank_portfolio3.csv', # Change name of output
                   input_rank_01_name = r'../recommender_system_leads/predicts/CL_rank_03.csv', # Change name of input CLassification script
                   input_rank_02_name = r'../recommender_system_leads/predicts/LR_rank_03.csv', # Change name of input Logistic Regression script
                   input_rank_03_name = r'../recommender_system_leads/predicts/RF_rank_03.csv'): # Change name of input Random Forest script

    # Read inputs:
    input_rank_01=pd.read_csv(input_rank_01_name)
    input_rank_02=pd.read_csv(input_rank_02_name)
    input_rank_03=pd.read_csv(input_rank_03_name)

    # Create auxs:
    aux_01=input_rank_01[input_rank_01.id.isin(input_rank_02.id)].id
    aux_02=input_rank_03[input_rank_03.id.isin(aux_01)].id

    # Treat rank 01 predictions:
    out_rank_01=input_rank_01[input_rank_01.id.isin(aux_02)]
    out_rank_01=out_rank_01.sort_values(by=['id'])
    out_rank_01.set_index(['id'],inplace=True)
    out_rank_01['rank']=out_rank_01['prob'].rank(ascending=False,method='min')
    out_rank_01=out_rank_01.drop('prob',axis=1)

    # Treat rank 02 predictions:
    out_rank_02=input_rank_02[input_rank_02.id.isin(aux_02)]
    out_rank_02=out_rank_02.sort_values(by=['id'])
    out_rank_02.set_index(['id'],inplace=True)
    out_rank_02['rank']=out_rank_02['prob'].rank(ascending=False,method='min')
    out_rank_02=out_rank_02.drop('prob',axis=1)

    # Treat rank 03 predictions:
    out_rank_03=input_rank_03[input_rank_03.id.isin(aux_02)]
    out_rank_03=out_rank_03.sort_values(by=['id'])
    out_rank_03.set_index(['id'],inplace=True)
    out_rank_03['rank']=out_rank_03['prob'].rank(ascending=False,method='min')
    out_rank_03=out_rank_03.drop('prob',axis=1)

    # Create output DataFrame:
    output=pd.DataFrame(columns=['rank_01','rank_02','rank_03','rank_mean','rank_std'],index=out_rank_01.index)
    output.loc[:,'rank_01']=out_rank_01['rank']
    output.loc[:,'rank_02']=out_rank_02['rank']
    output.loc[:,'rank_03']=out_rank_03['rank']
    output.loc[:,'rank_mean']=round(output.mean(axis=1),3)
    output.loc[:,'rank_std']=round(output.std(axis=1),3)
    output=output.sort_values(by='rank_mean')

    # Complete output features and save as csv file:
    rec_lead=output[output['rank_mean']<=output['rank_mean'].quantile(q=quantile_leads)]
    rec_lead.to_csv(out_rank_name)

aggregate_rank()
