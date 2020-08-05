import json
import numpy as np
import pandas as pd
from m5_utils import reduce_mem_usage, set_float32
from config import AUX_PATH,EXT_HD,N_CORES
from config import ROLS_LAG_1, ROLS_LAG_2, ROLS_SPLIT_1, ROLS_SPLIT_28,ROLS_SNAP
import pickle
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")



# Read files
with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)

## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def non_zero(x):
    return np.count_nonzero(x)

def nanstd(x):
    return np.nanstd(x)

def tmp_lag(DAY):
    shift_day = DAY[0]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_lag_'+str(shift_day)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day))
    return tmp_df[[col_name]]

def tmp_nonzero_cnt(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_nonzero_cnt_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).apply(non_zero, engine='cython',raw=True))
    return tmp_df[[col_name]]

def tmp_cnt(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_cnt_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).count())
    return tmp_df[[col_name]]

def tmp_rstd2(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_std2_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df['proxy'] = tmp_df['demand'].apply(lambda x: x if x!=0 else np.nan)
    tmp_df[col_name] = tmp_df.groupby(['id'])['proxy'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).apply(nanstd, engine='cython',raw=True))
    return tmp_df[[col_name]]

def tmp_std(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_std_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).std())
    return tmp_df[[col_name]]

def tmp_sum(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_sum_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).sum())
    return tmp_df[[col_name]]

def tmp_mean(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','demand']]
    col_name = 'tmp_mean_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return tmp_df[[col_name]]

def tmp_snap(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = data[['id','date','snap']]
    col_name = 'tmp_snap_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = data.groupby(['id'])['snap'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return tmp_df[[col_name]]

######################################################################33

## Set function, label and ROLS
f = tmp_nonzero_cnt
lbl = 'tmp_nonzero_cnt'
ROLS = ROLS_SPLIT_28

def create_lag_rolling(f, lbl, ROLS):

    for store_id in tqdm(range(0,10)):
        data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
        data.sort_values(by=['id','date'], ascending=True, inplace=True)
        data = data[['id','store_id','d','date','demand']]
        data = data[data['store_id']==store_id]
        data = data.reset_index(drop=True)
        del data['store_id']
        
        if lbl == 'tmp_lag':
            name1 = str(ROLS[0][0]) 
            name2 = str(ROLS[-1][0])
        else:
            name1 = str(ROLS[0][0]) +'_' +str(ROLS[0][1])
            name2 = str(ROLS[-1][0]) +'_' +str(ROLS[-1][1])


        tmp = df_parallelize_run( f, ROLS)
        data = pd.concat([data[['id','d']],tmp], axis=1)
        pickle.dump(data , open('data/'+lbl+ '_'+ name1 +'_'+ name2 +'_store_id_'+str(store_id)+ '.pkl', 'wb'))
        if store_id ==0:
            print(data.columns.tolist())
