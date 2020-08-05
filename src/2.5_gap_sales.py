import os
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import AUX_PATH
import argparse
import warnings
warnings.filterwarnings("ignore")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-u', '--upper_limit', required=True, default='2016-03-28',
	help='max date used to calculate the encoding, format YYYY-MM-DD')
ap.add_argument('-l', '--lower_limit', required=False, default='2013-03-01',
	help='min date used to calculate the encoding, format YYYY-MM-DD')
args = vars(ap.parse_args())


MAX_DATE = args['upper_limit']
MIN_DATE = args['lower_limit']
filename = str(MAX_DATE)
if MIN_DATE != '2013-03-01':
    filename = str(MAX_DATE) + '_custom'

if not os.path.exists('data/grid_df_'+filename+ '.pkl'):

    data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
    data['date'] = pd.to_datetime(data['date'])

    # Enconding will be calculating between the MIN_DATE and MAX_DATE
    data['demand'][data['date']>=MAX_DATE] = np.nan
    data['demand'][data['date']<MIN_DATE] = np.nan
    data.dropna(axis=0, subset=['demand'], inplace= True)
    data.sort_values(by=['id','date'], ascending=True, inplace=True)
    data.reset_index(drop=True,inplace=True)

    base_cols = ['id','store_id','d','demand']
    grid_df = data[base_cols]

    # Note: in 'gap' column: 1 is a day without sales:
    grid_df['gaps'] = (~(grid_df['demand'] > 0)).astype(int)


    prods = list(grid_df.id.unique())
    s_list = [] #list to hold gaps in days
    e_list = [] #list to hold expected values of gaps
    p_list = [] #list to hold avg probability of no sales

    # magic x8 speed booster thanks to @nadare
    for prod_id, df in tqdm(grid_df.groupby("id")):
        
        # get total days of series
        total_days = len(df)
        
        # extract gap_series for a prod_id
        sales_gaps = df.loc[:,'gaps']

        # calculate initial probability
        zero_days = sum(sales_gaps)
        p = zero_days/total_days

        # find and mark gaps
        accum_add_prod = np.frompyfunc(lambda x, y: int((x+y)*y), 2, 1)
        sales_gaps[:] = accum_add_prod.accumulate(df["gaps"], dtype=np.object).astype(int)
        sales_gaps[sales_gaps < sales_gaps.shift(-1)] = np.NaN
        sales_gaps = sales_gaps.fillna(method="bfill").fillna(method='ffill')
        s_list += [sales_gaps]
        
        # calculate E/total_days for all possible gap lengths:
        gap_length = sales_gaps.unique()
        
        d = {length: ((1-p**length)/(p**length*(1-p)))/365 for length in gap_length}
        sales_E_years = sales_gaps.map(d)
        
        # cut out supply_gap days and run recursively
        p1 = 0
        while p1 < p:
            
            if p1!=0:
                p=p1
            
            # once in 100 years event; change to your taste here
            gap_days = sum(sales_E_years>100)
                
            p1 = (zero_days-gap_days+0.0001)/(total_days-gap_days)
            
            d = {length: ((1-p1**length)/(p1**length*(1-p1)))/365 for length in gap_length}
            sales_E_years = sales_gaps.map(d)
            
        # add results to list it turns out masked replacemnt is a very expensive operation in pandas, so better do it in one go
        e_list += [sales_E_years]
        p_list += [pd.Series(p,index=sales_gaps.index)]
        
    # add it to grid_df in one go fast!:
    grid_df['gap_days'] = pd.concat(s_list)
    grid_df['gap_e'] = pd.concat(e_list)
    grid_df['sale_prob'] = pd.concat(p_list)


    # becuase we have some really extreme values lets take a log:
    grid_df['gap_e_log10'] = np.log10((grid_df['gap_e'].values+1))

    # e over 100 years does not make much sense
    m = grid_df['gap_e_log10']>2
    grid_df.loc[m,'gap_e_log10']=2 


    # Dump to pickle:
    grid_df.to_pickle('data/grid_df_'+filename+'.pkl')


for store_id in tqdm(range(0,10)):
    data = pd.read_pickle(AUX_PATH + 'data_v1.pkl')
    data = data[data['store_id']==store_id]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by=['id','date'], ascending=True, inplace=True)
    data.reset_index(drop=True,inplace=True)
    data = data[['id','d']]
    
    grid_df = pd.read_pickle('data/grid_df_'+filename+'.pkl')
    grid_df = grid_df[grid_df['store_id']==store_id]
    del grid_df['store_id']
    grid_df = grid_df[['id','d','sale_prob','gap_e_log10']]
    
    data = pd.merge(data, grid_df, on=['id','d'],how='left')
    data['gap_e_log10_t28r30']  = data.groupby(['id'])['gap_e_log10'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['sale_prob_t28r30']  = data.groupby(['id'])['sale_prob'].transform(lambda x: x.shift(28).rolling(30).mean())

    data.to_pickle('data/gap_sales_'+filename+'_store_id_'+str(store_id)+'.pkl')