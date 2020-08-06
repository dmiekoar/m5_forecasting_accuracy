import pandas as pd
import datetime as dt
from dateutil.relativedelta import *

class TimeBasedCV(object):
    '''
    Parameters 
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days
    '''
    
    
    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

        
        
    def split(self, data, validation_split_date=None, date_column='record_date', gap=0, tdelta=180):
        '''
        Generate indices to split data into training and test set
        
        Parameters 
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date 
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets
        
        Returns 
        -------
        train_index ,test_index: 
            list of tuples (train index, test index) similar to sklearn model selection
        '''
        
        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)
                    
        train_indices_list = []
        test_indices_list = []

        if validation_split_date==None:
            validation_split_date = data[date_column].min().date() + dt.timedelta(days=self.train_period)
        
        validation_split_date = pd.to_datetime(validation_split_date)
        
        start_train = validation_split_date - dt.timedelta(days=self.train_period)
        end_train = start_train + dt.timedelta(days=self.train_period)- dt.timedelta(days=1)
        start_test = end_train + dt.timedelta(days=gap) + dt.timedelta(days=1)
        end_test = start_test + dt.timedelta(days=self.test_period)- dt.timedelta(days=1)

        while end_test <= data[date_column].max().date():
            #print('\nwithin while')
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date>=start_train) & 
                                     (data[date_column].dt.date<=end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date>=start_test) &
                                    (data[date_column].dt.date<=end_test)].index)
            
            #print("Train period:",start_train," to " , end_train,"# train records", len(cur_train_indices))
            #print("Test period: ",start_test, " to " , end_test ,"# test records", len(cur_test_indices))
            #print('\n')
            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + dt.timedelta(days=self.test_period) + dt.timedelta(days=tdelta) 
            end_train = end_test
            start_test = end_train + dt.timedelta(days=gap)+ dt.timedelta(days=1)
            end_test = start_test + dt.timedelta(days=self.test_period)- dt.timedelta(days=1)



        # mimic sklearn output  
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]

        self.n_splits = len(index_output)
        
        return index_output
    
    
    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits 