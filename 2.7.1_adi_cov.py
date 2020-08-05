import numpy as np
import pandas as pd
from tqdm import tqdm
from config import AUX_PATH,EXT_HD
from config import ROLS_SPLIT_28
import warnings
warnings.filterwarnings("ignore")

ROLS = ROLS_SPLIT_28
    
name1 = str(ROLS[0][0]) +'_' +str(ROLS[0][1])
name2 = str(ROLS[-1][0]) +'_' +str(ROLS[-1][1])


df_name1 = 'tmp_cnt'
df_name2 = 'tmp_nonzero_cnt'
df_name3 = 'tmp_sum'
df_name4 = 'tmp_rstd2'
roll_period = name1
roll_period2 = name2

for store_id in tqdm(range(0,10)):
    df1 = pd.read_pickle(EXT_HD + df_name1 + '_' + name1 +'_'+ name2 + '_store_id_/'+ df_name1 + '_' + name1 +'_'+ name2 + '_store_id_'+str(store_id) +'.pkl'); print(df1.shape)

    df2 = pd.read_pickle(EXT_HD + df_name2 + '_' + name1 +'_'+ name2 + '_store_id_/'+ df_name2 + '_' + name1 +'_'+ name2 + '_store_id_'+str(store_id) +'.pkl'); print(df2.shape)

    df3 = pd.read_pickle(EXT_HD + df_name3 + '_' + name1 +'_'+ name2 + '_store_id_/'+ df_name3 + '_' + name1 +'_'+ name2 + '_store_id_'+str(store_id) +'.pkl'); print(df3.shape)

    df4 = pd.read_pickle(EXT_HD + df_name4 + '_' + name1 +'_'+ name2 + '_store_id_/'+ df_name4 + '_' + name1 +'_'+ name2 + '_store_id_'+str(store_id) +'.pkl'); print(df4.shape)
    del df2['id'],df2['d']
    del df3['id'],df3['d']
    del df4['id'],df4['d']

    df = pd.concat([df1,df2,df3,df4], axis=1); del df1,df2,df3,df4;df

    for ls in ROLS:
        a = ls[0]
        b = ls[1]
        roll_period = str(a) +'_' +str(b)
        df['tmp_adi_'+ roll_period] = df['tmp_cnt_'+ roll_period]/df['tmp_nonzero_cnt_'+ roll_period]
        df['tmp_cov_den_'+ roll_period] = df ['tmp_sum_'+ roll_period]/df['tmp_nonzero_cnt_'+ roll_period]
        df['tmp_cov_'+ roll_period] = np.square(df['tmp_std2_'+ roll_period]/df['tmp_cov_den_'+ roll_period])
        del df['tmp_cov_den_'+ roll_period]
        del df['tmp_cnt_' + roll_period]
        del df['tmp_nonzero_cnt_' + roll_period]
        del df['tmp_sum_' + roll_period]
        del df['tmp_std2_' + roll_period]
    
    df.to_pickle('data/tmp_adi_cov_'+name1+'_'+name2+'_store_'+str(store_id)+ '.pkl')

