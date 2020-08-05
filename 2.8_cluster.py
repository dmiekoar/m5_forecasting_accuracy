
import numpy as np 
import pandas as pd
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
from scipy.cluster.hierarchy import linkage, fcluster

with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)
with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)


print('Cluster feature') 
sales_catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
sales_numcols = [f"d_{day}" for day in range(1, 1942)]
sales_dtypes = {numcol:"float32" for numcol in sales_numcols} 
sales_dtypes.update({col: "category" for col in sales_catcols if col != "id"})

id_cols = ['store_id', 'item_id']
sales_cols = ['d_'+str(i) for i in range(1,1942)]
cluster_df = pd.read_csv('input/sales_train_evaluation.csv',usecols=id_cols+sales_cols, index_col=id_cols) \
                            .astype(np.uint16).sort_index()
cluster_df = cluster_df.reset_index()

STORES_IDS = cluster_df['store_id']
STORES_IDS = list(STORES_IDS.unique())

for store_id in STORES_IDS:


    if store_id == 'TX_2':

        df_cluster2 = cluster_df.loc[cluster_df.store_id == store_id] \
                                        .assign(d_median=lambda x: x.median(axis=1)) \
                                        .query('d_median >= 1') \
                                        .drop(columns='d_median') \
                                        .iloc[:, -28*24:] 

        Z = linkage((df_cluster2>0).astype(int), method='ward')
        clust = fcluster(Z, t=11, criterion='maxclust')
        df_cluster2['cluster'] = clust
        cluster_df.loc[cluster_df.index.isin(df_cluster2.index), 'cluster'] = df_cluster2['cluster'].values
        
    else:

        df_cluster2 = cluster_df.loc[cluster_df.store_id == store_id] \
                                        .assign(d_median=lambda x: x.median(axis=1)) \
                                        .query('d_median >= 1') \
                                        .drop(columns='d_median') \
                                        .iloc[:, -28*24:] 

        Z = linkage((df_cluster2>0).astype(int), method='ward')
        clust = fcluster(Z, t=10, criterion='maxclust')
        df_cluster2['cluster'] = clust
        cluster_df.loc[cluster_df.index.isin(df_cluster2.index), 'cluster'] = df_cluster2['cluster'].values

        
cluster_df['cluster'] = cluster_df['cluster'].fillna(0)    
cluster_df['cluster'] = cluster_df['cluster'].astype(np.int16)
cluster_df = cluster_df[['store_id','item_id','cluster']]
cluster_df

cluster_df['id'] = cluster_df['item_id'] +'_' +cluster_df['store_id']

cluster_df['id'] = cluster_df['id'].map(dict_m5_inv['id'])
cluster_df['store_id'] = cluster_df['store_id'].map(dict_m5_inv['store_id'])
del cluster_df['item_id']
cluster_df.to_pickle('cluster_evaluation.pkl')
cluster_df