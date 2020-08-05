import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
from m5_utils import reduce_mem_usage
from config import AUX_PATH,EXT_HD
from config import ROLS_SPLIT_28
import warnings
warnings.filterwarnings("ignore")


def find_last_sale(df,n_day):
    
    # Limit initial df
    ls_df = df[['id','d','demand']]
    
    # Convert 'demand' to binary
    ls_df['non_zero'] = (ls_df['demand']>0).astype(np.int8)
    
    # Make lags to prevent any leakage
    ls_df['non_zero_lag'] = ls_df.groupby(['id'])['non_zero'].transform(lambda x: x.shift(n_day).rolling(2000,1).sum()).fillna(-1)

    temp_df = ls_df[['id','d','non_zero_lag']].drop_duplicates(subset=['id','non_zero_lag'])
    temp_df.columns = ['id','d_min','non_zero_lag']

    ls_df = ls_df.merge(temp_df, on=['id','non_zero_lag'], how='left')
    ls_df['last_sale'] = ls_df['d'] - ls_df['d_min']

    return ls_df



############################################################
#######  TO USE FOR UPDATE DURING PREDICTION STEP ########
def update_last_sale(df, END_TRAIN, PREDICT_DAY):
    
    df['tmp_demand_lag_1'] = df['demand'].transform(lambda x: x.shift(1))
    df['tmp_last_sale_lag_1'] = df['last_sale'].transform(lambda x: x.shift(1))
    
    df.loc[df['d_int']==(END_TRAIN + PREDICT_DAY),'last_sale'] = df.loc[df['d_int']==(END_TRAIN + PREDICT_DAY), ['tmp_demand_lag_1','tmp_last_sale_lag_1']].apply(lambda x:update_ls_df(x[0],x[1]), axis=1)
    del df['tmp_demand_lag_1']; del df['tmp_last_sale_lag_1']
    
    return df

def update_ls_df( demand, last_sale, threshold=0.5):    
    if demand < threshold:
        return last_sale + 1
    else:
        return 0
###########################################################


# Create feature
if not os.path.exists('data/last_sale_df.pkl'):

    data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
    data['d'] = data['d'].apply(lambda x: x.split('_')[1]).astype(int)
    last_sale_df = data[['id','d','date','store_id','demand']]
    last_sale_df = reduce_mem_usage(last_sale_df)

    to_add = find_last_sale(last_sale_df,1)
    to_add = reduce_mem_usage(to_add)
    
    # Find last non zero
    # Need some "dances" to fit in memory limit with groupers
    last_sale_df = pd.merge(last_sale_df[['id','d','store_id']],to_add[['id','d','last_sale']], on=['id','d'], how='left')
    last_sale_df = reduce_mem_usage(last_sale_df)
    
    last_sale_df['d'] = last_sale_df['d'].apply(lambda x: 'd_'+str(x) )
    
    last_sale_df.to_pickle('data/last_sale_df.pkl')

# Split by store
for store_id in tqdm(range(0,10)):

    data = pd.read_pickle('data/last_sale_df.pkl')
    data = data[data['store_id']==store_id]
    data.reset_index(drop=True,inplace=True)
    del data['store_id']
    data.to_pickle('data/last_sale_df_store_id_'+str(store_id)+'.pkl')

