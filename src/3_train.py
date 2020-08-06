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



for store_id in range(10):

    features = [
        'item_id',
        'dept_id',
        'cat_id',
        ##'state_id',
        ##'demand',
        ##'date',
        ##'d',
        #'sell_price',
        ##'total_sales',
        'wday', 'month', 'year', 'weekend', 'day', 'week', 'period', 'week_month', 'pre_holiday',
        'enc_cat_id_mean', 'enc_cat_id_std', 'enc_dept_id_mean', 'enc_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std',
        'last_sale',
        'price_max', 'price_min', 'price_std', 'price_mean', 'price_norm', 
        'price_nunique',
        'item_nunique',
        'price_momentum', 'price_momentum_m', 'price_momentum_y',
        'sell_price_ratio',
        'sell_price_2d',
        'last_price_change',
        'release', 'wk_from_release',
        'gap_e_log10_t28r30',
        'sale_prob_t28r30',
        'cluster',
        'tmp_lag_28', 'tmp_lag_29', 'tmp_lag_30', 'tmp_lag_31', 'tmp_lag_32', 'tmp_lag_33', 'tmp_lag_34',
        'tmp_lag_35', 'tmp_lag_36', 'tmp_lag_37', 'tmp_lag_38', 'tmp_lag_39', 'tmp_lag_40', 'tmp_lag_41', 'tmp_lag_42',
        'tmp_mean_28_7', 'tmp_mean_28_14', 'tmp_mean_28_30', 'tmp_mean_28_60', 'tmp_mean_28_180',
        'tmp_mean_1_7', 'tmp_mean_1_14', 'tmp_mean_1_30', 'tmp_mean_1_60', 'tmp_mean_7_7', 'tmp_mean_7_14', 'tmp_mean_7_30', 'tmp_mean_7_60',
        'tmp_mean_14_7', 'tmp_mean_14_14', 'tmp_mean_14_30', 'tmp_mean_14_60',
        'tmp_std_28_7', 'tmp_std_28_14', 'tmp_std_28_30', 'tmp_std_28_60', 'tmp_std_28_180',
        #'tmp_adi_28_7', 'tmp_cov_28_7',
        'tmp_adi_28_14',  'tmp_adi_28_30', 'tmp_adi_28_60', 'tmp_adi_28_180',
        #'tmp_cov_28_14', 'tmp_cov_28_30', 'tmp_cov_28_60', 'tmp_cov_28_180',
        'event_name_1', #'event_name_2','event_type_1', 'event_type_2',
        'snap',
        'state_wkd','store_wkd',
        'state_cat_wkd', 'store_cat_wkd',
        'event',
        'state_cat_ev', 'store_cat_ev',
        'state_dept_ev', 'store_dept_ev',
        'state_ev', 'store_ev',
        'state_mean', 'state_dept_mean', 'state_cat_mean',
        'state_snap', 'store_snap',
        'state_cat_snap', 'store_cat_snap'
    ]

    categorical_feat = [col for col in features if col in all_categorical]

    filename = 'features/features_'+'v'+str(version)+'_'+str(store_id)+'_week_'+str(week)+'.json'
    with open(filename,'w') as json_file:
        json.dump(features,json_file)
        
    print('Features saved at ', filename)

    # Prepare/ Load dataset for a given store(store_id)
    _, _, _, _, data_train, _ = prepare_store_dataset( store_id,load=load, save_test=save_test,DATA_TYPE=DATA_TYPE)

    if LOG_NEPTUNE == 1:
        import neptune
        neptune.init('daniaragaki/m5-acc')
        neptune.create_experiment(name='store '+str(store_id),
                                params = params,
                                tags = [DATA_TYPE])
        def neptune_monitor():
            def callback(env):
                for name, loss_name, loss_value, _ in env.evaluation_result_list:
                    neptune.send_metric('{}_{}_fold_{}'.format(name, loss_name,fold_), \
                                        x=env.iteration, y=loss_value)
            return callback

    feature_importance_df = pd.DataFrame()

    folds = TimeBasedCV(train_period=TRAIN_PERIOD, test_period=TEST_PERIOD, freq='days')

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data_train[['date']], validation_split_date=VAL_SPLIT_DATE, \
                                                        date_column='date', gap=0, tdelta=TDELTA)):
        print('Fold:',fold_)
        print('Train data: ',data_train.iloc[trn_idx]['date'].min(), ' to ',data_train.iloc[trn_idx]['date'].max(), \
            ' #', len(trn_idx))
        print('Valid data: ',data_train.iloc[val_idx]['date'].min(), ' to ',data_train.iloc[val_idx]['date'].max(), \
            ' #', len(val_idx))
        
        data_train = reduce_mem_usage(data_train)
        trn_x, trn_y = data_train.iloc[trn_idx,:][features], data_train.iloc[trn_idx]['demand']
        val_x, val_y   = data_train.iloc[val_idx,:][features], data_train.iloc[val_idx]['demand']
        
        train_data = lgb.Dataset(trn_x, label=trn_y, feature_name=features, categorical_feature=categorical_feat,free_raw_data=True)
        train_data.raw_data = None
        
        valid_data = lgb.Dataset(val_x, label=val_y,   feature_name=features, categorical_feature=categorical_feat,free_raw_data=True) 
        valid_data.raw_data = None
        del trn_x,trn_y,val_x,val_y
        gc.collect()
        
        seed_everything(SEED)
        model_lgb = lgb.train(params,
                            train_data,
                            valid_sets = [train_data, valid_data],
                            callbacks=[neptune_monitor()],
                            verbose_eval = 100
                            )
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model_lgb.feature_importance()
        fold_importance_df["fold"] = fold_  
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        model_lgb.save_model('models/model_lgb_'+str(store_id)+'_week_'+str(week)+'_fold_'+str(fold_)+'_v'+str(version)+'.pkl')
        
    del data_train
    if LOG_NEPTUNE==1:
        neptune.send_text('features', str(features))
        neptune.stop()