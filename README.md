# M5 Forecast Accuracy

This repo contains the description and source code developed during the competition, held by Kaggle from 2nd March until 30th June, 2020.

M5 is the latest of the M competitions and the goal is to forecast the sales for the upcoming 28 days for each of 42.480 time series of the competition, using a dataset provided by Walmart. 

One of the main characteristics of this competition that makes it different from the previous ones is that many series display an intermittent behaviour.

For more details please visit [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/) and [MOFC](https://mofc.unic.ac.cy/m5-competition/).

---

This solution ranked 28th out of 5558 teams, placing it at the top 1%. You can find details of my proposed solution below, as well as code at [src](https://github.com/dmiekoar/m5_forecasting_accuracy/tree/master/src) folder.

#### 1. Pre Processing
In this section I reshape the dataset so we can draw insights more easily as well as start shaping it into a format we could feed into our model.

The complete dataset contains data from 2011 to 2016, a total of 1969 days for each of the 10 stores with a portfolio of 3049 products. This means there are more than 60 million rows.

Many variables are categorical so I've encoded them and created a dictionary to keep track of their actual values/meaning, while reducing the memory consumption.

Here we also merge `sell_prices` and `calendar` which can help us creating more features, and fix some possible mistakes and/or outliers I have come across during the exploratory analysis.

#### 2. Exploratory Analysis
Exploratory Analysis - still need organize the code, but many very good exploratory kernels were published on kaggle and contributed to this work.

#### 3. Feature Engineering
In this section we extract features related to date, sell price, events and snap. We also perform mean enconding on the target variable `demand` using different variables, used clustering to generate a new feature calculated ADI(average interval demand), and other features that could help take into account the intermmitency of our dataset.

#### 4. Machine Learning Model
##### 4.1. Training
I used lightgbm to create the model with a time-based cross validation. Not the whole available dataset was used for training, but only data from 2013 and on.
Training was performed for each store, as the exploratory analysis showed us stores did not always displayed the same pattern.

##### 4.2. Prediction
A recursive strategy to predict the sales was adopted. This means that instead of feeding the whole test dataset and predict `demand`, we have predicted the sales for each one of the 28 days, one by one, and for each new day we recalculate some of our features taking into account the result of the previous day.

#### 5. Post Processing and Submission
No post processing such as 'magical number' was applied to adjust the forecast.

The submission file was based only on a simple forecast average of the first and last folds result.

#### Notes
Pending - I intend to release and add the original solution I was working at once I'm able to finish it and compare the results.
