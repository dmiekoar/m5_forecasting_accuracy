import numpy as np
import pandas as pd
import json
from m5_utils import create_validation_submission
from model_config import version, DATA_TYPE


with open('input/dict_m5.json') as json_file:
    dict_m5 = json.load(json_file)
with open('input/dict_m5_inv.json') as json_file:
    dict_m5_inv = json.load(json_file)

predictions_validation = create_validation_submission('input')

preds_fold_0 = pd.read_csv('predictions/preds_v'+str(version)+'_' +DATA_TYPE+'_fold_0.csv')
preds_fold_1 = pd.read_csv('predictions/preds_v'+str(version)+'_' +DATA_TYPE+'_fold_1.csv')
preds_fold_2 = pd.read_csv('predictions/preds_v'+str(version)+'_' +DATA_TYPE+'_fold_2.csv')
preds_fold_3 = pd.read_csv('predictions/preds_v'+str(version)+'_' +DATA_TYPE+'_fold_3.csv')

test = preds_fold_0.copy()
test['id'] = test['id'].astype('str').map(dict_m5['id'])
test['id'] = test['id'] + '_evaluation'

df_submission = pd.concat([predictions_validation, test])
df_submission.to_csv('df_submission.csv', index=False)
df_submission


##########################################################################################
## Post processing - zeros ###############################################################

# base_test = pd.read_pickle('predictions/base_test_v'+str(version)+'_' +DATA_TYPE+'_fold_'+str(0)+'.pkl')

# base_test = base_test[['id','demand','date','d','tmp_mean_28_30','tmp_mean_28_60','tmp_mean_28_180']]
# base_test['id'] = base_test['id'].astype('str').map(dict_m5['id'])
# base_test['id'] = base_test['id'] + '_evaluation'

# zeros_lst=base_test[(base_test['tmp_mean_28_60']==0)&(base_test['date']=='2016-05-22')]['id'].tolist()

# zeros = test[~test['id'].isin(zeros_lst)]

# for col in  ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
#        'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
#        'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28']:
#     zeros[col] = zeros[col].apply(lambda x: x if x>0.5 else 0)

# to_add = zeros.copy()


# df_submission_evaluation = pd.concat([zeros, to_add])
# df_submission_evaluation.sort_values(by=['id'],ascending=True, inplace=True)
# final_submission = pd.concat([predictions_validation, df_submission_evaluation])
# final_submission.to_csv('df_submission.csv', index=False)