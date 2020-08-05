import numpy as np 
import pandas as pd
import pickle
import json
import warnings, os
warnings.filterwarnings('ignore')
from config import bucket, prefix, SUPPORT_PATH


# Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



# Read data(pickle) from s3 or local
def read_file(filename, file_type, source):
    import boto3

    if source == 's3':
        cred = boto3.Session().get_credentials()
        ACCESS_KEY = cred.access_key
        SECRET_KEY = cred.secret_key
        SESSION_TOKEN = cred.token  ## optional

        s3client = boto3.client('s3', 
                        aws_access_key_id = ACCESS_KEY, 
                        aws_secret_access_key = SECRET_KEY, 
                        aws_session_token = SESSION_TOKEN
                       )

        response = s3client.get_object(Bucket=bucket, Key=prefix + filename)

        body = response['Body'].read()
        df = pickle.loads(body)
    elif source != 'local':
            df = pickle.load(open(source + filename, 'rb'))
    else:
        if file_type == 'df':
            df = pd.read_pickle('input/' + filename)
        elif file_type == 'json':
            df = json.load(open('input/' + filename, 'rb'))
        else:
            df = pickle.load(open('input/' + filename, 'rb'))
    return df


def save_file(file, filename, file_type, bucket, prefix):
    import boto3

    if file_type != 'model':
        pickle.dump(file, open('input/' + filename, 'wb'))
        boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'input/' + filename)).upload_file('input/' + filename)
    else:
        pickle.dump(file, open('models/' + filename, 'wb'))
        boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'models/' + filename)).upload_file('models/' + filename)
        
# Show amount and percentage of missing values
def check_missing(dataset, display = 5):
    
    temp_df = dataset.copy()
    df_nan = (temp_df.isnull().sum() / len(temp_df)) * 100
    missing_data = pd.DataFrame({'Missing n': temp_df.isnull().sum(),'% Missing' :df_nan})
    if missing_data['Missing n'].sum() == 0:
        return print('Great! There are no missing values in this dataset.')
    else:
        return missing_data.sort_values('% Missing', ascending = False).head(display)

def create_validation_submission(ROOT_PATH):
    data = pd.read_csv(os.path.join(ROOT_PATH,'sales_train_evaluation.csv'))
    validation_dates = ['id'] +['d_'+ str(x) for x in range(1914,1942)]
    data = data[validation_dates]
    data.columns = ['id'] +['F'+ str(x) for x in range(1,29)]
    data['id'] = data['id'].transform(lambda x: x.replace('_evaluation','_validation'))
    data.to_csv('data/validation_submission.csv', index=False)
    return data


def set_float32(df):
    col = df.select_dtypes(np.float16).columns
    df[col] = df[col].astype(np.float32)
    return df

def get_new_columns(name, aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
