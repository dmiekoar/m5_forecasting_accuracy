
import os
import json
import gc
import datetime as dt
import numpy as np
import pandas as pd
from m5_utils import reduce_mem_usage, set_float32
from config import INPUT_PATH,EXT_HD
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

def get_price_features_1(df):

    df = set_float32(df)
    
    # We can do some basic aggregations
    df['price_max'] = df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    df['price_min'] = df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    df['price_std'] = df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    df['price_mean'] = df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    df['price_norm'] = df['sell_price']/df['price_max']

    # Some items are can be inflation dependent
    # and some items are very "stable"
    df['price_nunique'] = df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    df['item_nunique'] = df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

    del df['sell_price'], df['item_id']
    
    return df

def get_price_features_2(df):

    # Now we can add price "momentum" (some sort of)
    # Shifted by week 
    # by month mean
    # by year mean
    df['price_momentum'] = df['sell_price']/df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
    df['price_momentum_m'] = df['sell_price']/df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    df['price_momentum_y'] = df['sell_price']/df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

    del df['month'], df['year'], df['sell_price'], df['item_id']

    return df   

def get_price_features_3(df):
    
    df['sell_price_2d'] = df['sell_price'].round(2).astype(str)
    df['sell_price_2d'] = df['sell_price_2d'].apply(lambda x: str(x).split('.')[1])
    df['sell_price_2d'] = df['sell_price_2d'].apply(lambda x: (x).ljust(2,'0'))
    df['sell_price_1d'] = df['sell_price_2d'].apply(lambda x: x[-1:])

    return df



def consolidate_snap(df):
    
    df['snap'] = 0
    df.loc[df['state_id']==0,'snap'] = df.loc[df['state_id']==0,'snap_CA']
    df.loc[df['state_id']==1,'snap'] = df.loc[df['state_id']==1,'snap_TX']
    df.loc[df['state_id']==2,'snap'] = df.loc[df['state_id']==2,'snap_WI']

    del df['snap_CA'], df['snap_TX'], df['snap_WI']
    del df['state_id']
   
    return df


from math import ceil

def get_period(x):
    if x<=10:
        return 1
    elif x>=23:
        return 2
    else:
        return 0
    
    
def add_datetime_features(df):
    
    df['date'] = pd.to_datetime(df['date'])
    df['weekend'] = (df['wday'] <3).astype(int)
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.week
    df['period'] = df['day'].apply(lambda x: get_period(x))
    df['week_month'] = df['day'].apply(lambda x: ceil(x/7)).astype(np.int8)
    
    return data

def release_period(sell_prices):
    
    release_period = sell_prices.groupby(['item_id','store_id'])['wm_yr_wk'].min()
    release_period = release_period.reset_index(drop = False)
    release_period.columns = ['item_id','store_id','release']

    return release_period

def add_first_date(df,release_period ):
    
    df = df.merge(release_period, on=['item_id','store_id'], how = 'left')
    
    df['release'] = df['release'] - df['release'].min()
    df['release'] = df['release'].astype(np.int16)
    
    return df

def release(data, sell_prices):
    
    release = release_period(sell_prices)
    data = add_first_date(data, release)
    return data


def transform(df):
    
    print('Removing NAN sell_price')
    df.dropna(axis=0, subset=['sell_price'], inplace= True)

    return df
##############################################################

with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)


###################################################################
### Consolidate snap into one single column #######################

data = pd.read_pickle('data/data_v0.2.pkl')
data = data[['id', 'store_id', 'state_id', 'd',  'snap_CA', 'snap_TX','snap_WI']]
data = consolidate_snap(data)
data.to_pickle('data/data_snap.pkl')
data.head().append(data.tail())

#####################################################################
### Create price features 1 #########################################

for store_id in tqdm(range(0,10)):

    data = pd.read_pickle('data/data_full.pkl')
    data = data[['id','store_id','item_id', 'd','sell_price']]
    data.sort_values(by=['id'], ascending=True, inplace=True)
    data = data[data['store_id']==store_id]
    data = data.reset_index(drop=True)
    data = get_price_features_1(data)
    data = data[['id', 'store_id', 'd', 'price_max', 'price_min', 'price_std','price_mean', 'price_norm', 'price_nunique', 'item_nunique']]
    del data['store_id']
    pickle.dump(data , open('data/price_fe_1_store_id_'+str(store_id)+ '.pkl', 'wb'))
    
data.head().append(data.tail())

#####################################################################
### Create price features 2 #########################################

data = pd.read_pickle('data/data_full.pkl')
data = data[['id','store_id','item_id', 'd','sell_price','month','year']]
data = get_price_features_2(data)

data.to_pickle('data/price_fe_2.pkl')
data.head().append(data.tail())

#####################################################################
### Create price features 3 #########################################

data = pd.read_pickle('data/data_v1.pkl')
data = data[['id','store_id', 'd','sell_price']]
data = get_price_features_3(data)

data.to_pickle('data/price_fe_3.pkl')
data.head().append(data.tail())

#####################################################################
####### Create price ratio relative to first price ##################

tmp = pd.read_pickle('data/data_v1.pkl')
tmp.sort_values(by=['id','date'],ascending=True, inplace=True)
tmp.drop_duplicates(subset=['id'], keep='first',inplace=True)
tmp['first_price'] = tmp['sell_price']
tmp = tmp[['id','first_price']]

data = pd.read_pickle('data/data_v1.pkl')
data = data[['id','d','store_id','sell_price']]
data = data.merge(tmp, on=['id'], how='left')
data['sell_price_ratio'] = data['sell_price']/data['first_price']
del data['sell_price'], data['first_price']
data.to_pickle('data/price_ratio.pkl')

#####################################################################
######### Create date features ######################################

data = pd.read_pickle('data/data_v0.2.pkl')
data = data[['id','item_id', 'store_id','date', 'wday', 'month', 'year', 'd','wm_yr_wk']]
sell_prices = convert_numbers(sell_prices, dict_m5_inv)
data = pd.merge(data,sell_prices, on=['item_id','store_id','wm_yr_wk'], how='left')
del data['wm_yr_wk'], data['item_id'], sell_prices
data = transform(data)
del data['sell_price']
data.to_pickle('data/data_tmp_1.pkl')

#--------------------------------------------------------------------

data = pd.read_pickle('data/data_tmp_1.pkl')
data = add_datetime_features(data)
data = reduce_mem_usage(data)
data.to_pickle('data/date_fe.pkl')

#####################################################################
######### Add some date columns to snap file ########################

data = pd.read_pickle('data/data_v1.pkl')

add_data = pd.read_pickle('data/data_snap.pkl') 
add_data = add_data[['id','d','snap']]
data = pd.merge(data, add_data, on=['id','d'], how='left')

add_data = pd.read_pickle('data/date_fe.pkl')
add_data = add_data[['id','d','weekend','week_month']]
data = pd.merge(data, add_data, on=['id','d'], how='left')

data.to_pickle('data/date_snap.pkl')

#####################################################################
######### Obtain release data ######################################

data = pd.read_pickle('data/data_v0.2.pkl')
data = data[['id','item_id', 'store_id','date' ,'d','wm_yr_wk']]
sell_prices = convert_numbers(sell_prices, dict_m5_inv)
data = pd.merge(data,sell_prices, on=['item_id','store_id','wm_yr_wk'], how='left')
data = transform(data)
data = data[['id', 'item_id', 'store_id','date', 'wm_yr_wk', 'd']]
data = release(data, sell_prices)
data['wk_from_release'] = data['wm_yr_wk']-data['release']-11101
del data['item_id'],data['wm_yr_wk']
data.to_pickle('data/release.pkl')

#####################################################################
### Create dataset only with the dates ##############################
##### when product wasnt available -- using in metric ###############

# data = pd.read_pickle('data/data_full_v0.pkl')
# data = data[data['sell_price'].isna()==True]
# data.to_pickle('data/non_active.pkl')

#####################################################################
### Event features - no changes #####################################

data = pd.read_pickle('data/data_v0.2.pkl')
data = data[['id','store_id','d','event_name_1','event_name_2','event_type_1','event_type_2']]
data.to_pickle('data/events.pkl')

####################################################################
### Merge event and preholidays into a single variable #############

# Join pre holiday and events into event
data = pd.read_pickle('data/data_v1.pkl')

add_data = pd.read_pickle(EXT_HD+'_base_files/events.pkl')
add_data = add_data[['id','d','event_name_1']]
data = pd.merge(data, add_data, on=['id','d'], how='left')

# Preholidays is generated separately 
add_data = pd.read_pickle(EXT_HD+'_base_files/preholidays.pkl')
add_data = add_data[['id','d','pre_holiday']]
data = pd.merge(data, add_data, on=['id','d'], how='left')

del data['sell_price'], data['total_sales']

data['event_name_1'] = data['event_name_1'].cat.codes
data['event'] = data['event_name_1'] + data['pre_holiday']
data['event'] = (data['event']>=0).astype(int)
data.to_pickle('data/event_preholiday.pkl')
