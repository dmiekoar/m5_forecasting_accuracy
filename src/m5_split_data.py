import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from config import ROOT_PATH,AUX_PATH,EXT_HD
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 100)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename', required=True,
	help='filename, without extension')
args = vars(ap.parse_args())

filename = args['filename']

data = pd.read_pickle('data/'+ filename + '.pkl')
print(data.columns)

for store_id in tqdm(range(0,10)):
    data = pd.read_pickle('data/' + filename + '.pkl')
    data.sort_values(by=['id'], ascending=True, inplace=True)
    data = data[data['store_id']==store_id]
    data = data.reset_index(drop=True)
    del data['store_id']
    if store_id ==0:
        print(data.head().append(data.tail()))
    pickle.dump(data , open('data/'+filename+'_store_id_'+str(store_id)+ '.pkl', 'wb'))
