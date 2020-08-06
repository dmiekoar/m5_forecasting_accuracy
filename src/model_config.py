DATA_TYPE = 'evaluation'


LOG_NEPTUNE = True
NUM_ITEMS = 3049
DAYS_PRED = 28

SEED = 42
version = '0'

load = True
save_test = False

USE_CUSTOM_FEATURES = True
week = 0
store_id = 0
stores = [0,1,2,3,4,5,6,7,8,9]


if DATA_TYPE == 'validation':

    END_TRAIN = '2016-04-24'
    BEGIN_VAL = '2016-03-28'
    END_VAL = '2016-04-24'
    BEGIN_TEST = '2016-01-16'
    BEGIN_TEST_ds = '2016-04-25'
    END_TEST = '2016-05-22' 
    BEGIN_W1 = '2016-03-28'
    BEGIN_W2 = '2016-04-04'
    BEGIN_W3 = '2016-04-11'
    BEGIN_W4 = '2016-04-18'
    END_W1 = '2016-04-03'
    END_W2 = '2016-04-10'
    END_W3 = '2016-04-17'
    END_W4 = '2016-04-24'
    VAL_SPLIT_DATE ='2016-02-29'
    TRAIN_PERIOD = 1070
    TEST_PERIOD = 14
    TDELTA = 205

elif DATA_TYPE == 'evaluation':

    END_TRAIN = '2016-05-22'
    BEGIN_VAL = '2016-04-25'
    END_VAL = '2016-05-22'
    BEGIN_TEST = '2016-01-16'
    BEGIN_TEST_ds = '2016-05-23'
    END_TEST = '2016-06-19'
    BEGIN_W1 = '2016-04-25'
    BEGIN_W2 = '2016-05-02'
    BEGIN_W3 = '2016-05-09'
    BEGIN_W4 = '2016-05-16'
    END_W1 = '2016-05-01'
    END_W2 = '2016-05-08'
    END_W3 = '2016-05-15'
    END_W4 = '2016-05-22'
    VAL_SPLIT_DATE ='2016-03-28'
    TRAIN_PERIOD = 1070+28
    TEST_PERIOD = 14
    TDELTA = 205


# List with all categorical features
all_categorical = ['item_id', 'dept_id', 'cat_id', \
    'wday', 'month', 'year', 'weekend', 'day', 'quarter', 'week', 'wm_yr_wk', 'period', 'week_month', 'pre_holiday', \
    'sell_price_2d', 'sell_price_2d_cat', 'sell_price_1d', \
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'event', 'snap', 'cluster']


###### MODEL PARAMETERS ####################

params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.52,
                    'subsample_freq': 1,
                    'learning_rate': 0.025,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 50,
                    'n_estimators': 1300,
                    'lambda_l1': 0,
                    'lambda_l2': 0,
                    'boost_from_average': False,
                    'verbose': -1,
                    'seed': SEED
}
