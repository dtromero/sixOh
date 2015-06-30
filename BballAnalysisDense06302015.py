# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#==============================================================================
# To do list
#==============================================================================
# (1) Train an SVM model
# (2) Use the countvectorizer and sparse matrices to incorporate a much larger
# subset
# (3) Incorporate rolling previous day, 5 day, 10 day and 30 day avg stats
#==============================================================================
#%% Import libraries and data
#==============================================================================
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

engine = create_engine('sqlite:///E:/Bball/bballStats.db')

df = pd.read_sql('select * from gameDaysStats',
                 engine)
                 
df.date = pd.to_datetime(df.date)           # coerce date/times
df = df.dropna(subset=['FanDuelPts'],axis=0)# drop all na rows
df.ranker = df.ranker.astype('int')         # convert ranker to integer

df1 = df[df.date > '1995-01-01']            # only use records post 1994
df1 = df1.sort('date',ascending=False)      # sort by date

del df                                      # del dataframe for memory savings
#==============================================================================
#%%  Reduce the number of rows to prototype quicker
#==============================================================================
df1 = df[df.date > '2012-01-01']    # This must be done before deleting df,
                                    # you can simply subset df1 on itself
#%% Let's Look at rolling averages

df1['FanDuelPts_PrevGame'] = df1.groupby('player').FanDuelPts.apply(lambda x:\
                                                            x.shift(-1))
df1['FanDuelPts_Last3']  = df1.groupby('player').FanDuelPts_PrevDay.apply(\
    pd.rolling_mean, window=3)
df1['FanDuelPts_Last10']  = df1.groupby('player').FanDuelPts_PrevDay.apply(\
    pd.rolling_mean, window=10)
df1['FanDuelPts_Last30']  = df1.groupby('player').FanDuelPts_PrevDay.apply(\
    pd.rolling_mean, window=30)
#==============================================================================
#==============================================================================
# Predict last two weeks
#==============================================================================
#==============================================================================
#==============================================================================
# %%Reshape dataframe by converting all binary columns to n number of columns 
#for n number of unique factors in each categorical variable.

#We`start by creating these expanded dataframes for each categorical variable,
# then we merge them all together for our final feature matrix.
#==============================================================================
# get_dummies expands the categorical variable into a dataframe of columns
# where each column is a unique factor in that variable
dfplayer = pd.get_dummies(df1.player)
dfteamid = pd.get_dummies(df1.team_id)
dfloc    = pd.get_dummies(df1.game_location)
dfopp    = pd.get_dummies(df1.opp_id)

# Our feature set is composed of the above factorized variables and the below
# list of numeric/datetime variables from the original dataset
keep_cols = ['ranker','FanDuelPts','date']

df2 = pd.merge(df1.reset_index()[keep_cols], dfplayer.reset_index(),
         left_index=True, right_index=True)
df3 = pd.merge(df2.reset_index(), dfteamid.reset_index(),
         left_index=True, right_index=True)
df4 = pd.merge(df3.reset_index(), dfloc.reset_index(),
         left_index=True, right_index=True)
df5 = pd.merge(df4.reset_index(), dfopp.reset_index(),
         left_index=True, right_index=True)
         
# Drop extraneous columns created on the joins and delete the temporary 
# dataframes
df5 = df5.drop(['index_x','index_x','level_0'], axis=1)     
del df2, df3, df4, dfplayer, dfteamid, dfloc, dfopp

#==============================================================================
# %%Sort by train and test sets
#==============================================================================

#date1 = '2015-04-01'
date1 = '2014-11-15'    # Date separating test and train sets

# X is your feature space, Y is your target variable
dftrain_X = df5[df5.date < date1]
dftest_X  = df5[df5.date >=date1]
dftrain_y = df5[df5.date < date1]['FanDuelPts']
dftest_y  = df5[df5.date >=date1]['FanDuelPts']

dftrain_X = dftrain_X.drop(['date','FanDuelPts'],axis=1)
dftest_X  = dftest_X.drop(['date','FanDuelPts'],axis=1)

#==============================================================================
# %%Predict last two weeks
#==============================================================================

from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor(verbose=True)   # Parameters need to be tuned
rf1.fit_transform(dftrain_X,dftrain_y)      # Train the model
rf1_preds = rf1.predict(dftest_X)           # Predict against the test set

# Performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(dftest_y,rf1_preds)     
np.sqrt(mean_squared_error(dftest_y,rf1_preds))

#%% Plot model results
dfplot = pd.merge(dftest_y.to_frame('Actual'), pd.DataFrame(rf1_preds,
                  columns=['Pred']), left_index=True, right_index=True)
#%% Plot model results
dfplot.plot()

import matplotlib.pyplot as plt
plt.plot(np.arange(1,len(rf1_preds),),rf1_preds)

#%% SVM Takes forever to train
from sklearn.svm import SVR

svr1 = SVR(verbose=True)
svr1.fit(dftrain_X,dftrain_y)
svr1_preds = svr1.predict(dftest_X)

mean_absolute_error(dftest_y,svr1_preds)
np.sqrt(mean_squared_error(dftest_y,svr1_preds))

#==============================================================================
# %%Ideas
#==============================================================================
#add in time series analysis
#maybe start off by binning dates and regressing against those instead of 
#incorporating lag factors