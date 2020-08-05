import numpy as np 
import pandas as pd
import os
from m5_utils import reduce_mem_usage
from config import AUX_PATH
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def find_last_price_change(df,n_day):
    
    # Limit initial df
    ls_df = df[['id','d','sell_price']]
    ls_df['lag_1'] = ls_df.groupby('id')['sell_price'].transform(lambda x: x.shift(-1))
    
    ls_df['diff'] = ls_df['sell_price']-ls_df['lag_1']
    ls_df['non_zero'] = (ls_df['diff']!=0).astype(np.int8)
    
    # Make lags to prevent any leakage
    ls_df['non_zero_lag'] = ls_df.groupby(['id'])['non_zero'].transform(lambda x: x.shift(n_day).rolling(2000,1).sum()).fillna(-1)
    temp_df = ls_df[['id','d','non_zero_lag']].drop_duplicates(subset=['id','non_zero_lag'])
    temp_df.columns = ['id','d_min','non_zero_lag']

    ls_df = ls_df.merge(temp_df, on=['id','non_zero_lag'], how='left')
    ls_df['last_price_change'] = ls_df['d'] - ls_df['d_min']

    return ls_df

# Create feature
if not os.path.exists('data/last_price_change.pkl'):

    data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
    data['d'] = data['d'].apply(lambda x: x.split('_')[1]).astype(int)
    last_price_change_df = data[['id','d','date','store_id','sell_price']]
    last_price_change_df = reduce_mem_usage(last_price_change_df)

    to_add = find_last_price_change(last_price_change_df,1)
    to_add = reduce_mem_usage(to_add)
    
    # Find last non zero
    # Need some "dances" to fit in memory limit with groupers
    last_price_change_df = pd.merge(last_price_change_df[['id','d','store_id']],to_add[['id','d','last_price_change']], on=['id','d'], how='left')
    last_price_change_df = reduce_mem_usage(last_price_change_df)
    
    last_price_change_df['d'] = last_price_change_df['d'].apply(lambda x: 'd_'+str(x) )
    
    last_price_change_df.to_pickle('data/last_price_change.pkl')

# Split by store    
for store_id in tqdm(range(0,10)):

    data = pd.read_pickle('data/last_price_change.pkl')
    data = data[data['store_id']==store_id]
    data.reset_index(drop=True,inplace=True)
    del data['store_id']
    data.to_pickle('data/last_price_change_store_id_'+str(store_id)+'.pkl')

