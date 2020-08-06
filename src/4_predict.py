import os
import json
import sys
import gc
import math
import datetime as dt
import numpy as np
import pandas as pd
from m5_utils import *
from m5_prepare_store import *
from model_config import *
from config import *
from TimeBasedCV import TimeBasedCV
import gc, time, pickle, psutil, random
from itertools import product
from multiprocessing import Pool

from sklearn import preprocessing, metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


import warnings
warnings.filterwarnings("ignore")

with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)

def update_last_sale(df, END_TRAIN, PREDICT_DAY):
    
    df['tmp_demand_lag_1'] = df['demand'].transform(lambda x: x.shift(1))
    df['tmp_last_sale_lag_1'] = df['last_sale'].transform(lambda x: x.shift(1))
    
    df.loc[df['date']==(END_TRAIN + dt.timedelta(days=PREDICT_DAY)),'last_sale'] = \
    df.loc[df['date']==(END_TRAIN + dt.timedelta(days=PREDICT_DAY)), ['tmp_demand_lag_1','tmp_last_sale_lag_1']].apply(lambda x:update_ls_df(x[0],x[1]), axis=1)
    del df['tmp_demand_lag_1']; del df['tmp_last_sale_lag_1']
    
    return df

def update_latest_demand(df, END_TRAIN, PREDICT_DAY):
    
    
    df.loc[df['date']==(END_TRAIN + dt.timedelta(days=PREDICT_DAY)),'demand'] = \
    df.loc[df['date']==(END_TRAIN + dt.timedelta(days=PREDICT_DAY)), 'demand'].apply(lambda x:update_demand(x))
    
    return df

def update_ls_df( demand, last_sale, threshold=0.5):    
    if demand < threshold:
        return last_sale + 1
    else:
        return 0
    
def update_demand( demand, threshold=0.5):    
    if demand < threshold:
        return 0
    elif demand<=1:
        return 1
    else:
        return demand
    
def non_zero(x):
    return np.count_nonzero(x)

def nanstd(x):
    return np.nanstd(x)#np.nanstd(np.where(np.isclose(x,0), np.nan, x))
    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df
        
def tmp_nonzero_cnt(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_nonzero_cnt_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).apply(non_zero, engine='cython',raw=True))
    return tmp_df[[col_name]]

def tmp_cnt(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_cnt_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).count())
    return tmp_df[[col_name]]

def tmp_rstd2(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_std2_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df['proxy'] = tmp_df['demand'].apply(lambda x: x if x!=0 else np.nan)
    tmp_df[col_name] = tmp_df.groupby(['id'])['proxy'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).apply(nanstd, engine='cython',raw=True))
    return tmp_df[[col_name]]

def tmp_std(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_std_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).std())
    return tmp_df[[col_name]]

def tmp_sum(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_sum_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).sum())
    return tmp_df[[col_name]]

def tmp_mean(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','demand']]
    col_name = 'tmp_mean_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['demand'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return tmp_df[[col_name]]

def tmp_snap(DAY):
    shift_day = DAY[0]
    roll_wind = DAY[1]
    tmp_df = base_test[['id','date','snap']]
    col_name = 'tmp_snap_'+str(shift_day)+'_'+str(roll_wind)
    tmp_df[col_name] = tmp_df.groupby(['id'])['snap'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return tmp_df[[col_name]]

def replace_store(df, col_format_id, store_id, path_baseline):
    
    # Depending if the 'id column is numeric or text, obtain 'store_id'
    if col_format_id == 'numeric':
        df['id']  = df['id'].astype(str).map(dict_m5['id'])
        df['store_id']  = df['id'].map(dict_m5['id_to_store_str'])
        df['store_id']  = df['store_id'].map(dict_m5['store_id'])
    elif col_format_id == 'text':
        df['store_id']  = df['id'].map(dict_m5['id_to_store_id'])
        df['store_id']  = df['store_id'].map(dict_m5['store_id'])
    #print(df.head())
    
    # Create an empty dataframe that will be filled with data from the stores we want to evaluate
    chosen_stores = pd.DataFrame()
    for store in store_id: #store_id should be a list
        tmp = df[df['store_id']==store] 
        #print(tmp.head())
        chosen_stores = pd.concat([chosen_stores,tmp], axis=0)
    del chosen_stores['store_id']
    #print(chosen_stores.head())
    chosen_stores['id'] = chosen_stores['id'].str.replace('_evaluation','_validation')

    
    # Load the submission file that is the baseline and will have data replaced
    submission = pd.read_csv(path_baseline)
    # Add store_id to identify which rows will be replaced
    submission['store_id']  = submission['id'].map(dict_m5['id_to_store_str'])
    submission['store_id']  = submission['store_id'].map(dict_m5_inv['store_id'])
    submission['type'] = submission['id'].str.rpartition('_')[2]
    
    # Remove the current information from chosen stores
    for store in store_id:
        submission_idx = submission[(submission['store_id']==store)&(submission['type']=='validation')].index
        submission.drop(submission_idx, axis=0,inplace=True)
        submission.reset_index(drop=True,inplace=True)
    del submission['store_id']
    del submission['type']
    #print(submission.head())
    
    # Add the new information  --- fix id order
    submission = pd.concat([submission,chosen_stores], axis=0)
    submission.reset_index(drop=True, inplace=True)
    
    submission.to_csv('submission/submission_v'+str(version)+'.csv', index=False)
    
    return submission

def predict(test_evaluation, formatted = True):
    
    ### VALIDATION
    predictions_validation = create_validation_submission('input')

    
    ### EVALUATION
    if formatted == True:
        test_evaluation['id']  = test_evaluation['id'].astype(str).map(dict_m5['id'])
    else:
        test_evaluation = test_evaluation[(test_evaluation['date']>='2016-05-23')&(test['date']<='2016-06-19')]
        test_evaluation['id']  = test_evaluation['id'].astype(str).map(dict_m5['id'])

        predictions_evaluation = test_evaluation[['id', 'date', 'demand']]

        predictions_evaluation = pd.pivot(predictions_evaluation, index = 'id', columns = 'date', values = 'demand').reset_index()
        predictions_evaluation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    final = pd.concat([predictions_validation, predictions_evaluation])
    final.to_csv('submission/submission_v'+str(version)+'.csv', index = False)
    
    return final

def merge_test(stores =[0,1,2,3,4,5,6,7,8,9],DATA_TYPE='validation'):
    
    test = pd.DataFrame()
    for store_id in stores:
        print('Reading store_id: ', store_id)
        tmp = pd.read_pickle( AUX_PATH+'store_data/test_store_id_'+str(store_id)+'_'+DATA_TYPE+'.pkl')
        tmp['store_id'] = store_id
        test = pd.concat([test,tmp], axis=0)
    test.sort_values(by=['date','id'], ascending=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    test.to_pickle(AUX_PATH + 'store_data/test_' +DATA_TYPE+'_v'+str(version)+'.pkl')

    return test


test = merge_test(stores=stores,DATA_TYPE='validation' )

for fold_ in range(4):
    #fold_ = 0

    week = 0
    # Create DataFrame to store predictions
    all_preds = pd.DataFrame()

    # Join back the Test dataset with 
    # a small part of the training data 
    # to make recursive features
    base_test = pd.read_pickle(AUX_PATH + 'store_data/test_'+DATA_TYPE+ '_v'+str(version)+'.pkl')
    base_test['date'] = pd.to_datetime(base_test['date'])
    base_test.sort_values(by=['id','date'], ascending=True, inplace=True)
    base_test.reset_index(drop=True, inplace=True)

    base_test.loc[base_test['date']>=BEGIN_TEST_ds, 'last_sale'] = np.nan
    base_test = base_test[[col for col in base_test.columns if col not in drop_list]]

    END_TRAIN = pd.to_datetime(END_TRAIN)


    # Timer to measure predictions time 
    main_time = time.time()

    # Loop over each prediction day
    # As rolling lags are the most timeconsuming
    # we will calculate it for whole day
    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()
        
        ###################### RECALCULATING FEATURES ####################################
        ## Recalculating last_sale
        base_test = update_last_sale(base_test, END_TRAIN, PREDICT_DAY)
        #base_test = update_latest_demand(base_test, END_TRAIN, PREDICT_DAY)
        #print('Last sale feature updated')

        # Make temporary grid to calculate rolling lags
        grid_df = base_test.copy()

        #print('Recalculating features...')
        ###################### RECALCULATING FEATURES ####################################
        ## Recalculating rolling_mean_X_Y
        
        #print('make_lag_roll..')
        grid_df = pd.concat([grid_df, df_parallelize_run(tmp_mean, ROLS_SPLIT_1)], axis=1)

        store_id_ = 0 ## all features are the same among stores
        for store_id in stores: #STORES_IDS:
            
            #print('Loading model: ', store_id)
            # Read all our models and make predictions
            # for each day/store pairs        
            
            model_lgb = lgb.Booster(model_file='models/model_lgb_'+str(store_id)+'_week_0'+\
                                            '_fold_'+str(fold_)+'_v'+str(version)+'.pkl')
            MODEL_FEATURES = json.load(open('features/features_'+'v'+str(version)+'_'+str(store_id_)+'_week_0'+'.json'))

            day_mask = base_test['date']==( END_TRAIN + dt.timedelta(days=PREDICT_DAY))
            store_mask = base_test['store_id']==store_id
        
            mask = (day_mask)&(store_mask)
            base_test['demand'][mask] = model_lgb.predict(grid_df[mask][MODEL_FEATURES],predict_disable_shape_check=True)
        
        #base_test[day_mask] = update_latest_demand(base_test[day_mask], END_TRAIN, PREDICT_DAY)

        # Make good column naming and add 
        # to all_preds DataFrame
        temp_df = base_test[day_mask][['id','demand']]
        temp_df.columns = ['id','F'+str(PREDICT_DAY)]
        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=['id'], how='left')
        else:
            all_preds = temp_df.copy()
            
        print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                    ' %0.2f min total |' % ((time.time() - main_time) / 60),
                    ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
        del temp_df
        
        
    all_preds = all_preds.reset_index(drop=True)
    filename = 'predictions/preds_v'+str(version)+'_' +DATA_TYPE+'_fold_'+str(fold_)+'.csv'
    fn_base = 'predictions/base_test_v'+str(version)+'_' +DATA_TYPE+'_fold_'+str(fold_)+'.pkl'

    all_preds.to_csv(filename, index=False)
    base_test.to_pickle(fn_base)
    print('Saved file: ', filename)
    all_preds

print('Concluded')
