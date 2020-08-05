
import numpy as np
import pandas as pd
from m5_utils import reduce_mem_usage, set_float32
from config import AUX_PATH, ROOT_PATH,EXT_HD
import gc
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



####################################################################
### encoding events ############################################

data = pd.read_pickle('data/event_preholiday.pkl')
data['demand'][data['date']>='2016-04-25'] = np.nan
data['demand'][data['date']<'2013-03-01'] = np.nan
data.sort_values(by=['id','date'],ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)

del data['event_name_1'], data['pre_holiday']

data['state_cat_ev'] = data.groupby(['item_id','state_id','cat_id','event'])['demand'].transform('mean')
data['store_cat_ev'] = data.groupby(['item_id','store_id','cat_id','event'])['demand'].transform('mean')
del data['cat_id']
data = reduce_mem_usage(data);gc.collect()

data['state_dept_ev'] = data.groupby(['item_id','state_id','dept_id','event'])['demand'].transform('mean')
data['store_dept_ev'] = data.groupby(['item_id','store_id','dept_id','event'])['demand'].transform('mean')
del data['dept_id']
data = reduce_mem_usage(data);gc.collect()

data['state_ev'] = data.groupby(['item_id','state_id','event'])['demand'].transform('mean')
data['store_ev'] = data.groupby(['item_id','store_id','event'])['demand'].transform('mean')

data = reduce_mem_usage(data);gc.collect()
del data['item_id'], data['demand'], data['date'], data['state_id']
data.to_pickle('data/enc_event_holiday.pkl')

####################################################################

data = pd.read_pickle('data/date_snap.pkl')
data['demand'][data['date']>='2016-04-25'] = np.nan
data['demand'][data['date']<'2013-03-01'] = np.nan
data.sort_values(by=['id','date'],ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)
data['state_snap'] = data.groupby(['item_id','state_id','snap'])['demand'].transform('mean')
data['store_snap'] = data.groupby(['item_id','store_id','snap'])['demand'].transform('mean')

data = reduce_mem_usage(data);gc.collect()

data['state_cat_snap'] = data.groupby(['item_id','state_id','cat_id','snap'])['demand'].transform('mean')
data['store_cat_snap'] = data.groupby(['item_id','store_id','cat_id','snap'])['demand'].transform('mean')
del data['snap']
data = reduce_mem_usage(data);gc.collect()


data = data[['id', 'store_id',  'd','state_snap', 'store_snap', 'state_cat_snap', 'store_cat_snap']]
data.to_pickle('data/enc_snap.pkl')

####################################################################

data = pd.read_pickle('data/date_snap.pkl')
data['demand'][data['date']>='2016-04-25'] = np.nan
data['demand'][data['date']<'2013-01-01'] = np.nan
data.sort_values(by=['id','date'],ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)

data['state_wkd'] = data.groupby(['item_id','state_id','weekend'])['demand'].transform('mean')
data['store_wkd'] = data.groupby(['item_id','store_id','weekend'])['demand'].transform('mean')
data = reduce_mem_usage(data);gc.collect()
data['state_cat_wkd'] = data.groupby(['item_id','state_id','cat_id','weekend'])['demand'].transform('mean')
data['store_cat_wkd'] = data.groupby(['item_id','store_id','cat_id','weekend'])['demand'].transform('mean')
data = reduce_mem_usage(data);gc.collect()
data = data[['id', 'store_id', 'd', 'state_wkd', 'store_wkd', 'state_cat_wkd','store_cat_wkd']]
data.to_pickle('data/enc_date.pkl')

####################################################################

data = pd.read_pickle('data/data_v1.pkl')
data['demand'][data['date']>='2016-04-25'] = np.nan
data['demand'][data['date']<'2013-01-01'] = np.nan
data.sort_values(by=['id','date'],ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)
data['state_mean'] = data.groupby(['item_id','state_id'])['demand'].transform('mean')
data['state_dept_mean'] = data.groupby(['item_id','state_id','dept_id'])['demand'].transform('mean')
data['state_cat_mean'] = data.groupby(['item_id','state_id','cat_id'])['demand'].transform('mean')
data = reduce_mem_usage(data);gc.collect()
data = data[['id', 'store_id',  'd', 'state_mean','state_dept_mean', 'state_cat_mean']]
data.to_pickle('data/enc_state.pkl')

####################################################################
