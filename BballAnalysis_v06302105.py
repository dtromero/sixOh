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
# (4) Add rolling average stats/FPs
#==============================================================================
# Import libraries and data
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
#  Reduce the number of rows to prototype quicker
#==============================================================================
df1 = df[df.date > '2012-01-01']    # This must be done before deleting df,
                                    # you can simply subset df1 on itself

#==============================================================================
#==============================================================================
# Predict last two weeks
#==============================================================================
#==============================================================================
#==============================================================================
# Reshape dataframe by converting all binary columns to n number of columns 
#for n number of unique factors in each categorical variable.

#We`start by creating these expanded dataframes for each categorical variable,
# then we merge them all together for our final feature matrix.
#==============================================================================
# get_dummies expands the categorical variable into a dataframe of columns
# where each column is a unique factor in that variable

def dense2sparse(dfin):
    dfplayer = pd.get_dummies(dfin.player).to_sparse()
    dfteamid = pd.get_dummies(dfin.team_id).to_sparse()
    dfloc    = pd.get_dummies(dfin.game_location).to_sparse()
    dfopp    = pd.get_dummies(dfin.opp_id).to_sparse()
    
    # Our feature set is composed of the above factorized variables and the below
    # list of numeric/datetime variables from the original dataset
    keep_cols = ['ranker','FanDuelPts','date']
    keep_cols = ['ranker']
    
    #Convert dfin to a sparse matrix, reset the index because we're simply joining 
    # index upcoming/hstacking them.We only need to keep the above two columns.
    # We'll need to add date back later so we can arrange the test and train sets
    # by date. But right now, we can't convert datetime columns to the sparse
    # matrix format
    dfs = dfin[keep_cols].reset_index()[keep_cols].to_sparse() # Note: keep_cols is
    #      called twice to drop the index column after resetting the index
    
    # Import scipy sparse module for joining sparse matrices
    import scipy.sparse as sparse
    
    #horiztonally stack all sparse matrices
    df2 = sparse.hstack(blocks = [dfs, dfplayer, dfteamid, dfloc, dfopp])
    
    # Drop extraneous columns created on the joins and delete the temporary 
    # dataframes
    del dfplayer, dfteamid, dfloc, dfopp, keep_cols
    
    # Save list of feature names from sparse matrices
    dfsnames    = list(dfs.columns)
    players     = list(dfin.sort('player').player.unique())
    teams       = list(dfin.sort('team_id').team_id.unique())
    locations   = list(dfin.sort('game_location').game_location.unique())
    opponents   = list(dfin.sort('opp_id').opp_id.unique())
    
    # Create a list of the feature names for our sparse matrix
    feature_names = dfsnames + players + teams + locations + opponents
    return(df2, feature_names)

#==============================================================================
# Sort by train and test sets
#==============================================================================

#date1 = '2015-04-01'
date1 = '2014-11-15'    # Date separating test and train sets

#The dataframes are sorted by date, let's the first n rows after date 1
idxs  = df1[df1.date <= date1].index
idxs2   = df1[df1.date > date1].index

dftrain_Xdense = df1.ix[idxs]
dftest_Xdense = df1.ix[idxs2]

dftrain_X, feature_names = dense2sparse(dftrain_Xdense)
dftest_X,  feature_names = dense2sparse(dftest_Xdense)


# X is your feature space, Y is your target variable
dftrain_y = df1[df1.date <= date1]['FanDuelPts']
dftest_y  = df1[df1.date >  date1]['FanDuelPts']

#==============================================================================
# Predict last two weeks
#==============================================================================

from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor(verbose=True)   # Parameters need to be tuned
rf1.fit_transform(dftrain_X,dftrain_y)      # Train the model
rf1_preds = rf1.predict(dftest_X)           # Predict against the test set

# Performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(dftest_y,rf1_preds)     
np.sqrt(mean_squared_error(dftest_y,rf1_preds))

# Plot model results
dfplot = pd.merge(dftest_y.to_frame('Actual'), pd.DataFrame(rf1_preds,
                  columns=['Pred']), left_index=True, right_index=True)

dfplot.plot()

import matplotlib.pyplot as plt
plt.plot(np.arange(1,len(rf1_preds),),rf1_preds)

from sklearn.svm import SVR

svr1 = SVR(verbose=True)
svr1.fit(dftrain_X,dftrain_y)
svr1_preds = svr1.predict(dftest_X)

mean_absolute_error(dftest_y,svr1_preds)
np.sqrt(mean_squared_error(dftest_y,svr1_preds))

#==============================================================================
# Ideas
#==============================================================================
#add in time series analysis
#maybe start off by binning dates and regressing against those instead of 
#incorporating lag factors