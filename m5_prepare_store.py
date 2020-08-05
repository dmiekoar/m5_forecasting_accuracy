import numpy as np 
import pandas as pd
import warnings, os
import json
import random
import pickle
from m5_utils import read_file, set_float32, reduce_mem_usage
warnings.filterwarnings('ignore')

    
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
    