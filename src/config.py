import psutil

INPUT_PATH = 'input/'
ROOT_PATH = 'data/'
AUX_PATH = '/media/dani/MyBackup/Shared/m5_archive/auxiliar/'
EXT_HD = '/media/dani/SAMSUNG500/m5/updated_data/'
SUPPORT_PATH = '/media/dani/SAMSUNG500/m5/updated_data/_base_files/'

ARCHIVE_PATH = '/media/dani/MyBackup/Shared/m5_archive/'
LOCAL_PATH = 'data/'
EXT_HD = '/media/dani/SAMSUNG500/m5/updated_data/'
STORE_PATH = '/media/dani/MyBackup/Shared/m5_archive/store_data/'
STORE_PATH_local = '/media/dani/MyBackup/Shared/m5_archive/store_data/'


# AWS
bucket = 'mybucket-ir'                               
prefix = 'm5_forecasting_accuracy-master/'

MAX_DATE = '2016-04-25'
MIN_DATE = '2013-03-01'

N_CORES = psutil.cpu_count()

ROLS_LAG_1 = []
for i in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]:
        ROLS_LAG_1.append([i])
        
ROLS_LAG_2 = []
for i in [28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]:
        ROLS_LAG_2.append([i])
        
ROLS_SPLIT_1 = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT_1.append([i,j])    
        
ROLS_SPLIT_28 = []
for i in [28]:
    for j in [7,14,30,60,180]:
        ROLS_SPLIT_28.append([i,j])


ROLS_SNAP = ROLS_SPLIT_28