
import os
import json
import gc
import datetime as dt
import numpy as np
import pandas as pd
from m5_utils import reduce_mem_usage, set_float32
from config import INPUT_PATH
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

# Read files
calendar    = pd.read_csv(os.path.join(INPUT_PATH,'calendar.csv'));  del calendar['weekday']
data        = pd.read_csv(os.path.join(INPUT_PATH,'sales_train_evaluation.csv'))
submission  = pd.read_csv(os.path.join(INPUT_PATH,'sample_submission.csv'))
sell_prices = pd.read_csv(os.path.join(INPUT_PATH,'sell_prices.csv'))

def preprocess_sub(submission,generate_dict = None):
    
    # Identify ids used for validation and evaluation
    #test_rows = [row for row in submission['id'] if 'validation' in row]
    val_rows = [row for row in submission['id'] if 'evaluation' in row]
    
    # Create template for validation and evaluation        #sub_dict_1['test'] = (dict(zip(test_columns[1::],submission.columns[1::])))

    #test = submission[submission['id'].isin(test_rows)]
    val = submission[submission['id'].isin(val_rows)]
    # Remove '_evaluation' from 'id
    val['id'] = val['id'].transform(lambda x: x.rsplit('_evaluation')[0])

    # Identify which forecasting days belong to validation and evaluation
    n = 1914
    #test_days = np.arange(n, n+28, 1)
    val_days = np.arange(n+28, n+28+28, 1)
    #test_columns = ['id']+['d_'+ str(value) for value in test_days]
    val_columns = ['id']+['d_'+ str(value) for value in val_days]
    
    # Creates a dict to later be used as reference when submitting
    sub_dict_1 = {}
    if generate_dict is not None:
        sub_dict_1['val'] = (dict(zip(val_columns[1::],submission.columns[1::])))

    # Replace columns name
    #test.columns = test_columns
    val.columns = val_columns
    

    #test = pd.melt(test, id_vars= 'id', var_name= 'day', value_name= 'demand')
    val = pd.melt(val, id_vars= 'id', var_name= 'day', value_name= 'demand')
    
    return val, sub_dict_1 #test

def create_dict(train, calendar,sell_prices):
    # 1:'Saturday', 7:'Friday'
    
    sub_dict1 = {}
    # Define columns that will be in the dictionary
    cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    # Sort dataset by 'id'
    train.sort_values(by=['id'], ascending=True, inplace=True)
    # remove '_evaluation' from 'id'
    train['id'] = train['id'].transform(lambda x: x.rsplit('_evaluation')[0])
    
    # Loop through the train dataframe to get the values for the dictionary
    for col in train.columns:
        if col in cols:
            # Set the column as category and generate a dictionary out of it
            train[col] = train[col].astype('category')
            tmp_dict = dict(enumerate(train[col].cat.categories))
            # Compile the information into a nested dictionary
            sub_dict1[col] = tmp_dict

    # Define other columns that will be in the dictionary
    cols = [ 'wm_yr_wk','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    # Loop through the train dataframe to get the values for the dictionary
    for col in calendar.columns:
        if col in cols:
            # Set the column as category and generate a dictionary out of it
            calendar[col] = calendar[col].astype('category')
            tmp_dict = dict(enumerate(calendar[col].cat.categories))
            # Compile the information into a nested dictionary
            sub_dict1[col] = tmp_dict
    
    sell_prices['sell_price_2d'] = sell_prices['sell_price'].round(2).astype(str)
    sell_prices['sell_price_2d'] = sell_prices['sell_price_2d'].apply(lambda x: str(x).split('.')[1])
    sell_prices['sell_price_2d'] = sell_prices['sell_price_2d'].apply(lambda x: (x).ljust(2,'0'))
    sell_prices['sell_price_1d'] = sell_prices['sell_price_2d'].apply(lambda x: x[-1:])

    for col in sell_prices.columns:
        if col in ['sell_price_1d','sell_price_2d']:
            # Set the column as category and generate a dictionary out of it
            sell_prices[col] = sell_prices[col].astype('category')
            tmp_dict = dict(enumerate(sell_prices[col].cat.categories))
            # Compile the information into a nested dictionary
            sub_dict1[col] = tmp_dict
        
    
    keys = calendar['date'].tolist()
    values = calendar['d'].tolist()
    dictionary = dict(zip(keys, values))

    sub_dict1['d_date'] = dictionary
    sub_dict1['year'] = {1: 2011, 2: 2012, 3: 2013, 4: 2014, 5: 2015, 6: 2016}

    return train, calendar, sub_dict1

def create_inv_dict(source_dict):
    
    inv_dict = {}

    for key, value in source_dict.items():
        tmp_dict = {}
        for i, item in value.items():
            if key != 'year':
                tmp_dict[item] = i
        inv_dict[key] = tmp_dict
    
    inv_dict['year'] = {2011:1, 2012:2, 2013:3, 2014:4, 2015:5, 2016:6}

    return inv_dict

def preprocess(data, submission, calendar = calendar, sell_prices = sell_prices, generate_dict = None):
    
    if generate_dict is not None:
        data, calendar, dict_m5_1 = create_dict(data, calendar,sell_prices)
    else:
        dict_m5_1 = {}
    
    product = data[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    train = pd.melt(data, id_vars= data.iloc[:,0:6].columns, var_name= 'day', value_name= 'demand')
    train_columns = train.columns
        
    val, dict_m5_2 = preprocess_sub(submission, generate_dict)
    
    # Merge both dictionaries
    dict_m5 = dict_m5_1
    dict_m5.update(dict_m5_2)
        
    # Merge products to test and validation
    #test = test.merge(product, how = 'left', on = 'id')
    #val['id'] = val['id'].transform(lambda x: x.replace('_evaluation','_validation'))
    val = val.merge(product, how = 'left', on = 'id')
    #val['id'] = val['id'].transform(lambda x: x.replace('_validation','_evaluation'))

    #test = test[train_columns]
    val = val[train_columns]

    dict_m5_inv = create_inv_dict(dict_m5)
    

    del product, submission, data
    return train,  val, calendar, dict_m5, dict_m5_inv #test
    

def merge_df(df, calendar,sell_prices):
    
    calendar = set_float32(calendar)
    df = pd.merge(df, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    del df['day']
    #print('transform sell_price_32')
    #sell_prices = set_float32(sell_prices)
    #print('merge sell prices')
    #df = pd.merge(df, sell_prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'])
    #print('reduce memory')

    df = reduce_mem_usage(df)
        
    return df

def convert_numbers(df, dict_m5_inv):
    map_year = { 2011:1, 2012:2, 2013:3, 2014:4, 2015:5, 2016:6}

    for col in df.columns:
        if col not in ['demand', 'date','d','d_int' ,'wm_yr_wk','wday','month','year','snap_CA','snap_TX','snap_WI',\
                       'sell_price','sell_price_1d','sell_price_2d','total_sales','release',\
                      'price_max', 'price_min', 'price_std', 'price_mean','price_norm', 'price_nunique',\
                       'item_nunique', 'price_momentum','price_momentum_m', 'price_momentum_y']:
            col_dict = dict_m5_inv[col]
            df[col] = df[col].map(col_dict)
        elif col  in ['year']:
            df[col] = df[col].map(map_year)
        
    return df


def fix_christmas(df):

    df.loc[((df['date']=='2011-12-25')|(df['date']=='2012-12-25')|\
     (df['date']=='2013-12-25')|(df['date']=='2014-12-25')|\
     (df['date']=='2015-12-25')),'demand'] = np.nan
    df['demand']= df['demand'].ffill(axis=0)
    df['total_sales']= df['demand']*df['sell_price']

    return df

def transform(df):
    
    print('Removing NAN sell_price')
    df.dropna(axis=0, subset=['sell_price'], inplace= True)

    return df

def fix_prices(df):

    df.loc[df['sell_price']>100,'sell_price'] = 17.36
    df.loc[(df['sell_price']==61.46)&(df['item_id']==2938),'sell_price'] = 16.46

    return df


#############################################################################
## Melt dataset into format we can work with, and merge with additional data
## Create dictionary to help understanding data after enconding 

train, val, calendar, dict_m5, dict_m5_inv = preprocess(data, submission, calendar, sell_prices, generate_dict = True)
data = pd.concat([train, val], axis=0)
del train, val
data.to_pickle('data/data_v0.pkl')

with open('input/dict_m5.json', 'w') as json_file:
    json.dump(dict_m5, json_file)
with open('input/dict_m5_inv.json', 'w') as json_file:
    json.dump(dict_m5_inv, json_file)

data = merge_df(data, calendar,sell_prices)
data.to_pickle('data/data_v0.1.pkl')


###################################################################
### Convert data into numbers #####################################

with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)

# Convert merged data into numbers    
data = pd.read_pickle('data/data_v0.1.pkl')
data = convert_numbers(data, dict_m5_inv)
data.to_pickle('data/data_v0.2.pkl')
data.head().append(data.tail())


###################################################################
### Add sell_price ################################################
### Fix prices ####################################################
### Create base of full dataset ###################################

data = pd.read_pickle('data/data_v0.2.pkl')
data = data[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'demand',
       'date', 'wm_yr_wk', 'd','month','year']]
sell_prices = convert_numbers(sell_prices, dict_m5_inv)
data = pd.merge(data,sell_prices, on=['item_id','store_id','wm_yr_wk'], how='left')
data = fix_prices(data)
data.to_pickle('data/data_full.pkl')

#####################################################################
### Fix christmas ###################################################

data = pd.read_pickle('data/data_full.pkl')
data = data[['id','item_id','dept_id','cat_id','store_id','state_id','demand','date','d','sell_price']]
data = fix_christmas(data)
data.to_pickle('data/data_full_v0.pkl')

#####################################################################
### Remove registers where product wasnt available ##################

data = pd.read_pickle('data/data_full_v0.pkl')
data = transform(data)
data.sort_values(by=['id','date'], ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)
data.to_pickle('data/data_v1.pkl')

#####################################################################
### Add items to dictionary #########################################

data = pd.read_pickle('data/data_v0.pkl')
data.drop_duplicates(subset=['id'], keep='first', inplace=True)
data.reset_index(drop=True, inplace=True)


keys = data['id'].tolist()
values = data['store_id'].tolist()
dictionary = dict(zip(keys, values))
dict_m5['id_2_store_str'] = dictionary
dict_m5['id_2_store_str'] 

with open('input/dict_m5.json', 'w') as json_file:
    json.dump(dict_m5, json_file)
with open('input/dict_m5_inv.json', 'w') as json_file:
    json.dump(dict_m5_inv, json_file)

#####################################################################
### Add items to dictionary #########################################

data = pd.read_pickle( 'data/data_v1.pkl')
data.drop_duplicates(subset=['id'], keep='first', inplace=True)
data.reset_index(drop=True, inplace=True)

keys = data['id'].tolist()
values = data['store_id'].tolist()
dictionary = dict(zip(keys, values))
dict_m5['id_2_store'] = dictionary
dict_m5['id_2_store'] 

with open('input/dict_m5.json', 'w') as json_file:
    json.dump(dict_m5, json_file)
with open('input/dict_m5_inv.json', 'w') as json_file:
    json.dump(dict_m5_inv, json_file)





