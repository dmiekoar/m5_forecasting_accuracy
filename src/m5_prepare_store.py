import numpy as np 
import pandas as pd
import warnings, os
import json
import random
import pickle
from m5_utils import read_file, set_float32, reduce_mem_usage
from config import EXT_HD, STORE_PATH
from model_config import *
warnings.filterwarnings('ignore')

with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)
    
drop_list = []

drop_tmp_mean = ['tmp_mean_1_7', 'tmp_mean_1_14', 'tmp_mean_1_30','tmp_mean_1_60',\
                    'tmp_mean_7_7', 'tmp_mean_7_14', 'tmp_mean_7_30', 'tmp_mean_7_60', \
                    'tmp_mean_14_7', 'tmp_mean_14_14', 'tmp_mean_14_30', 'tmp_mean_14_60'
                    ]

drop_tmp_sum = ['tmp_sum_1_7', 'tmp_sum_1_14','tmp_sum_1_30', 'tmp_sum_1_60', \
                    'tmp_sum_7_7', 'tmp_sum_7_14',  'tmp_sum_7_30', 'tmp_sum_7_60', \
                    'tmp_sum_14_7', 'tmp_sum_14_14', 'tmp_sum_14_30', 'tmp_sum_14_60'  \
                ]

drop_tmp_std = ['tmp_std_1_7', 'tmp_std_1_14', 'tmp_std_1_30', 'tmp_std_1_60', \
                    'tmp_std_7_7', 'tmp_std_7_14',  'tmp_std_7_30', 'tmp_std_7_60', \
                    'tmp_std_14_7', 'tmp_std_14_14',  'tmp_std_14_30', 'tmp_std_14_60'
                ]

drop_tmp_nonzero_cnt = ['tmp_nonzero_cnt_1_7', 'tmp_nonzero_cnt_1_14', 'tmp_nonzero_cnt_1_30', 'tmp_nonzero_cnt_1_60',\
                    'tmp_nonzero_cnt_7_7', 'tmp_nonzero_cnt_7_14','tmp_nonzero_cnt_7_30', 'tmp_nonzero_cnt_7_60', \
                    'tmp_nonzero_cnt_14_7', 'tmp_nonzero_cnt_14_14', 'tmp_nonzero_cnt_14_30', 'tmp_nonzero_cnt_14_60' \
                        ]

drop_tmp_std2 = ['tmp_std2_1_7', 'tmp_std2_1_14', 'tmp_std2_1_30', 'tmp_std2_1_60', \
                    'tmp_std2_7_7', 'tmp_std2_7_14', 'tmp_std2_7_30', 'tmp_std2_7_60', \
                    'tmp_std2_14_7', 'tmp_std2_14_14',  'tmp_std2_14_30', 'tmp_std2_14_60'
                ]

drop_list = drop_tmp_mean
drop_list += drop_tmp_sum
drop_list += drop_tmp_nonzero_cnt
drop_list += drop_tmp_std
drop_list += drop_tmp_std2

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
def add_store_data(store_id,PATH=None):

    print('Loading item information...')
    data = read_file('item_data_store_id_/item_data_store_id_'+str(store_id)+'.pkl','df', PATH)
    
    return data


def add_features(data, store_id, filename, drop_list=None, PATH=None):
    
    print('Loading features from ', filename )
    tmp = read_file(filename +'/'+filename +str(store_id)+'.pkl','df', PATH); tmp = set_float32(tmp)
    if drop_list is not None:
        tmp.drop(drop_list, axis=1, inplace=True)
    lst_feat = list([col for col in tmp.columns if col not in ['id','d']])
    print('Adding features: ', lst_feat,'\n')
    data = pd.merge(data, tmp, on=['id','d'],how='left'); del tmp
    data = reduce_mem_usage(data)
    
    return data
    
    
def add_non_active(data,store_id,PATH=None):
    
    base = pickle.load(open(PATH +'non_active_store_id_/non_active_store_id_'+str(store_id)+ '.pkl', 'rb'))
    data = pd.concat([data,base], axis=0)
    
    return data


def prepare_store_dataset(store_id ,load=False, save_test=True, DATA_TYPE = 'validation'):
    
    if load==False:
        
        data = add_store_data(store_id, PATH =  EXT_HD)
                
        data = add_features(data, store_id, 'date_fe_store_id_', \
                            drop_list=['date'], PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'preholidays_store_id_', drop_list=None, PATH =  EXT_HD)
        
        if DATA_TYPE=='validation':
            data = add_features(data, store_id, 'mean_encoding_2016-03-28_store_id_', drop_list=None,PATH =  EXT_HD)

        elif DATA_TYPE =='evaluation':
            data = add_features(data, store_id, 'mean_encoding_2016-04-25_store_id_', drop_list=None,PATH =  EXT_HD)
            
        data = add_features(data, store_id, 'last_sale_df_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'price_fe_1_store_id_', drop_list=None, PATH =  EXT_HD)
        
        
        data = add_features(data, store_id, 'price_fe_2_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'price_fe_3_store_id_', drop_list=['sell_price_1d'], PATH =  EXT_HD)

        data = add_features(data, store_id, 'last_price_change_store_id_', drop_list=None, PATH =  EXT_HD)

        data = add_features(data, store_id, 'release_store_id_', drop_list=None, PATH =  EXT_HD)
        
        if DATA_TYPE=='validation':
            data = add_features(data, store_id, 'gap_sales_2016-03-28_store_id_', drop_list=['gap_e_log10', 'sale_prob'], PATH =  EXT_HD)

        elif DATA_TYPE =='evaluation':
            data = add_features(data, store_id, 'gap_sales_2016-04-25_store_id_', drop_list=['gap_e_log10', 'sale_prob'], PATH =  EXT_HD)

        tmp = read_file('cluster_'+DATA_TYPE+'.pkl','df', EXT_HD)
        data = pd.merge(data, tmp, on=['id'],how='left'); del tmp
                
        data = add_features(data, store_id, 'tmp_lag_28_48_store_id_', \
                            drop_list=['tmp_lag_43', 'tmp_lag_44', 'tmp_lag_45', 'tmp_lag_46', 'tmp_lag_47', 'tmp_lag_48'\
                                      ] , PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'tmp_mean_28_7_28_180_store_id_', drop_list=None , PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'tmp_mean_1_7_14_60_store_id_', drop_list=None , PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'tmp_std_28_7_28_180_store_id_', drop_list=None , PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'tmp_adi_cov_28_7_28_180_store_', drop_list=None, PATH =  EXT_HD)
              
        data = add_features(data, store_id, 'events_store_id_', drop_list=None, PATH =  EXT_HD)
                
        data = add_features(data, store_id, 'data_snap_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'enc_date_store_id_', drop_list=None, PATH =  EXT_HD)

        data = add_features(data, store_id, 'enc_event_holiday_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'enc_state_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data = add_features(data, store_id, 'enc_snap_store_id_', drop_list=None, PATH =  EXT_HD)

        data = add_features(data, store_id, 'price_ratio_store_id_', drop_list=None, PATH =  EXT_HD)
        
        data['sell_price_2d'] = data['sell_price_2d'].map(dict_m5_inv['sell_price_2d'])#.astype(int)
    
        cat_cols = data.select_dtypes('category')
        for col in cat_cols:
            print(col)
            data[col] = data[col].cat.codes
        
        print('Merges finished')
        data['store_id'] = store_id
        data = reduce_mem_usage(data)
        
        data.to_pickle(STORE_PATH + 'store_'+str(store_id)+'_'+DATA_TYPE +'.pkl')
    else:
        data = read_file('store_'+str(store_id)+'_'+DATA_TYPE +'.pkl','df',STORE_PATH)
    
        
    data.dropna(axis=0, subset=['sell_price'], inplace= True)

    
    del data['store_id']
    
    data_train  = data[data['date'] <= END_TRAIN ]
    data_test  = data[data['date'] >= BEGIN_TEST ]
    
    # drop_list = []

    # drop_tmp_mean = ['tmp_mean_1_7', 'tmp_mean_1_14', 'tmp_mean_1_30','tmp_mean_1_60',\
    #                   'tmp_mean_7_7', 'tmp_mean_7_14', 'tmp_mean_7_30', 'tmp_mean_7_60', \
    #                   'tmp_mean_14_7', 'tmp_mean_14_14', 'tmp_mean_14_30', 'tmp_mean_14_60'
    #                  ]
    
    # drop_tmp_sum = ['tmp_sum_1_7', 'tmp_sum_1_14','tmp_sum_1_30', 'tmp_sum_1_60', \
    #                   'tmp_sum_7_7', 'tmp_sum_7_14',  'tmp_sum_7_30', 'tmp_sum_7_60', \
    #                   'tmp_sum_14_7', 'tmp_sum_14_14', 'tmp_sum_14_30', 'tmp_sum_14_60'  \
    #                 ]
    
    # drop_tmp_std = ['tmp_std_1_7', 'tmp_std_1_14', 'tmp_std_1_30', 'tmp_std_1_60', \
    #                   'tmp_std_7_7', 'tmp_std_7_14',  'tmp_std_7_30', 'tmp_std_7_60', \
    #                   'tmp_std_14_7', 'tmp_std_14_14',  'tmp_std_14_30', 'tmp_std_14_60'
    #                ]
    
    # drop_tmp_nonzero_cnt = ['tmp_nonzero_cnt_1_7', 'tmp_nonzero_cnt_1_14', 'tmp_nonzero_cnt_1_30', 'tmp_nonzero_cnt_1_60',\
    #                   'tmp_nonzero_cnt_7_7', 'tmp_nonzero_cnt_7_14','tmp_nonzero_cnt_7_30', 'tmp_nonzero_cnt_7_60', \
    #                   'tmp_nonzero_cnt_14_7', 'tmp_nonzero_cnt_14_14', 'tmp_nonzero_cnt_14_30', 'tmp_nonzero_cnt_14_60' \
    #                        ]
    
    # drop_tmp_std2 = ['tmp_std2_1_7', 'tmp_std2_1_14', 'tmp_std2_1_30', 'tmp_std2_1_60', \
    #                   'tmp_std2_7_7', 'tmp_std2_7_14', 'tmp_std2_7_30', 'tmp_std2_7_60', \
    #                   'tmp_std2_14_7', 'tmp_std2_14_14',  'tmp_std2_14_30', 'tmp_std2_14_60'
    #                 ]
    
    # drop_list = drop_tmp_mean
    # drop_list += drop_tmp_sum
    # drop_list += drop_tmp_nonzero_cnt
    # drop_list += drop_tmp_std
    # drop_list += drop_tmp_std2
    
    to_drop = [col for col in data_test.columns if col in drop_list]
    
    data_test.drop(to_drop, axis = 1, inplace = True)
    
    del data
    
    if save_test:
        data_test.to_pickle(STORE_PATH +'test_store_id_'+str(store_id)+'_'+DATA_TYPE +'.pkl')

        
    data_train['date'] = pd.to_datetime(data_train['date'])
    data_train.sort_values(by=['date', 'id'], ascending=True,inplace=True)
    data_train.reset_index(drop=True,inplace=True)

    X_train = data_train[(data_train['date'] <= END_TRAIN)]
    y_train = X_train['demand']

    X_val = data_train[(data_train['date'] >= BEGIN_W1) & (data_train['date'] <= END_W4)]
    y_val = X_val['demand']
    
    return X_train, y_train, X_val, y_val, data_train, data_test
    