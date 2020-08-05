import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import AUX_PATH

import argparse
import warnings
warnings.filterwarnings("ignore")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-u', '--upper_limit', required=True,
	help='max date used to calculate the encoding, format YYYY-MM-DD')
ap.add_argument('-l', '--lower_limit', required=False, default='2013-03-01',
	help='min date used to calculate the encoding, format YYYY-MM-DD')
args = vars(ap.parse_args())



MAX_DATE = args['upper_limit']
MIN_DATE = args['lower_limit']
filename = str(MAX_DATE)
if MIN_DATE != '2013-03-01':
    filename = str(MAX_DATE) + '_custom'

for store_id in tqdm(range(0,10)):

    data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by=['id','date'], ascending=True, inplace=True)
    data = data[data['store_id']==store_id]
    data.reset_index(drop=True,inplace=True)


    # Enconding will be calculating between the MIN_DATE and MAX_DATE
    data['demand'][data['date']>=MAX_DATE] = np.nan
    data['demand'][data['date']<MIN_DATE] = np.nan


    base_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'date','d','demand']
    data = data[base_cols]

    cols =  ['cat_id','dept_id','item_id']

    for col in cols:
        data['enc_'+col +'_mean'] = data.groupby(col)['demand'].transform('mean').astype(np.float16)
        data['enc_'+col +'_std'] = data.groupby(col)['demand'].transform('std').astype(np.float16)   
        #data['proxy'] = data['demand'].apply(lambda x: x if x!=0 else np.nan)
        #data['enc'+col_name +'std_'] = data.groupby(col)['proxy'].transform(lambda x: np.nanstd(x)).astype(np.float16)
        #data['enc'+col_name +'nonzero_cnt'] = data.groupby(col)['demand'].transform(lambda x: np.count_nonzero(x)).astype(np.float16)

    #del data['proxy']

    keep_cols = [col for col in list(data) if col not in base_cols]
    grid_df = data[['id','d']+keep_cols]

    grid_df.to_pickle('data/mean_encoding_'+filename+'_store_id_'+str(store_id)+'.pkl')
    del grid_df, data
    gc.collect()