import datetime as dt
import numpy as np 
import pandas as pd
import pickle
from tqdm import tqdm
from config import AUX_PATH
import warnings
warnings.filterwarnings('ignore')

# Read calendar
filename = 'input/calendar.csv'
calendar = pd.read_csv(filename)
calendar = calendar[calendar['event_name_1'].isna()==False]
# Holidays we want to obtain pre holidays
holidays_list = [ "Mother's day", 'MemorialDay',"Father's day",'Easter']

calendar.loc[calendar['event_name_1'].isin(holidays_list)==False,'event_name_1'] = np.nan
calendar.loc[calendar['event_name_2'].isin(holidays_list)==False,'event_name_2'] = np.nan
calendar.loc[calendar['event_name_2'].isna()==False,'event_name_1'] = calendar['event_name_2']
# List of dates from holidays of interest
dates_list = calendar.loc[calendar['event_name_1'].isin(holidays_list)]['date'].tolist()


filename = 'data_v1'
data = pickle.load(open(AUX_PATH +filename+'.pkl', 'rb'))
data = data[['id','date','d','store_id']]
data['date'] = pd.to_datetime(data['date'])
data['pre_holiday'] = 0

# Set d-1, d-2 and d-3 as preholiday
for dates in dates_list:
    dates = pd.to_datetime(dates)
    data.loc[(data['date']==(dates-dt.timedelta(days=1))),'pre_holiday'] = 1
    data.loc[(data['date']==(dates-dt.timedelta(days=2))),'pre_holiday'] = 1
    data.loc[(data['date']==(dates-dt.timedelta(days=3))),'pre_holiday'] = 1

data.to_pickle('data/preholidays.pkl')

for store_id in tqdm(range(0,10)):
    tmp_data = data[data['store_id']==store_id]
    drop_list = ['date','store_id']
    tmp_data.drop(drop_list, axis=1, inplace=True)
    tmp_data.to_pickle('data/preholidays_store_id_'+str(store_id)+'.pkl')
