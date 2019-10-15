# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:05:51 2019

@author: 82091
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import OneClassSVM
import pickle
import os

file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

def load_data(file_path):
    '''
    Load dataset from filepath.
    '''
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print('file not found')
        
def random_shuffle(data, seed = 4):
    '''
    Randomly shuffle the entire dataset.
    '''
    np.random.seed(seed)
    new_indices = np.random.permutation(len(data), )
    data = data[new_indices]
    return data

def attribute_split(data):
    '''
    split index of dataset into categorical index and numerical index.
    
    Input[DataFrame]: dataset
    Output[dict]: {'cat': categorical index,
                  'num': numerical index}
    '''
    attr = {} # output
    column_types = data.dtypes
    # Numerical Index
    attr['num'] = column_types[(column_types==np.int64) | (column_types == np.float64)].index
    # Categorical Index
    attr['cat'] = column_types[column_types==np.object].index
    return attr
    
  
class data_selector(BaseEstimator, TransformerMixin):
    '''
    The objective of this transformer is to split data into numerical and categorical for building pipeline.
    select either numerical data or categorical data.
    When num_feature == True, select numerical data; when num_feature==False, select categorical data
    '''
    def __init__(self, num_feature = True):
        self.num_feature = num_feature     # defining which type of feature to select
    def fit(self, X, y=None):
        return self                        # doing nothing at fitting
    def transform(self, X, y=None):
        if self.num_feature:                # if num_feature is True, select numerical feature
            return X[attr['num']]
        else:                               # if num_feature is False, select categorical feature
            return X[attr['cat']]


class MyLabelEncoder(TransformerMixin):
    '''
    The pipeline is assuming LabelBinarizer's fit_transform method is defined 
    to take three positional arguments, while it is defined to take only two.
    An error will be raised, therefore we need customerize a labelencoder for building pipeline
    '''
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)
    def fit(self, x, y=0):                 # doing nothing at fitting
        return self
    def transform(self, x, y=0):
        '''
        return one hot encoding of a matrix rather than an array.
        '''
        return x.apply(self.encoder.fit_transform)  # apply customized encoder to every column


if __name__ == '__main__':
    df = load_data(file_name)
    df.info() # get the basic information about the dataset
    df.nunique() # check how many unique number or string there are in each row
    df.describe() # See basic statistics of numerical data, it can help us to see if there is extreme value
    df_corr = df.corr()
    sns.heatmap(df_corr)
    df.dtypes
    
    '''
    We found TotalCharges is categorical index while it should be numerical.
    The reason might be there is some empty space '\s' in some row, making pandas
    unable to convert it into float.
    
    Therefore, we apply regular expression to find out rows with empty space in it.
    '''
    df['TotalCharges'].str.match('[^0-9.]').sum() # check if there is any row with none-digit character
    df['TotalCharges'].str.match('\s').sum()     # check if there is any row with space
    df[df['TotalCharges'].str.match('\s')]['TotalCharges'] # See all the rows with space
    df[df['TotalCharges'].str.match('[^0-9.]')]['TotalCharges'] # See all the rows with none-digit
    df['TotalCharges'] = df['TotalCharges'].str.strip() # remove empty space from all rows
    df[df['TotalCharges']=='']['TotalCharges'] = np.nan
    
    
    for i in range(len(df)):
        try:
            float(df.loc[i,:]['TotalCharges'])
        except ValueError:
            print(i, df.loc[i]['TotalCharges'])
            df.loc[i]['TotalCharges'] = np.nan
            
    df.drop(df[df['TotalCharges']==''].index, inplace = True) # There is only 23 rows with missing value, therefore, we simply delete them
    df['TotalCharges'] = df['TotalCharges'].astype('float') # after removing all spaces, we can finally convert TotalCharges into float
    
    #################################### split data into dependent variable and independent variables
    Y = df['Churn'] 
    X = df.drop('Churn', axis = 1)
    X.drop('customerID', axis = 1, inplace=True)

    attr = attribute_split(X)            # call feature split function; split feature into categorical and numerical    
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3) # split data into training set and test set
  
    ###################################### Construct Pipeline 
    
    '''
    We build two pipelines here, 
    num_feature_transformer for transforming numerical   data
    cat_feature_transformer for transforming categorical data
    
    Then combine these two into a 
    feature_union pipeline
    '''
    
    
    num_feature_transformer = Pipeline([('num_selector', data_selector()),
                                        ('normalization', StandardScaler())])
    cat_feature_transformer = Pipeline([('cat_selector',data_selector(num_feature=False)),
                                        ('label_encoder',MyLabelEncoder()),
                                        ('one_hot_encoder', OneHotEncoder())])
    feature_union = FeatureUnion([('num_feature',num_feature_transformer),
                                  ('cat_feature',cat_feature_transformer)])
    
    x_train = feature_union.fit_transform(x_train)
    y_train = LabelEncoder().fit_transform(y_train)
    
    
    ##################################### Manifold Learning
    '''
    Apply manifold learning techniques to get intuition into data
    '''
    
    
    tSNE = TSNE(n_components = 2)
    low_x = tSNE.fit_transform(x_train.toarray())
    
    plt.figure(figsize = (20,16))
    plt.scatter(low_x[:,0], low_x[:,1], c = y_train)
    plt.savefig('t-sne.png')
    
    
    if not os.path.exists('low_x.pickle'):
        low_x_file = open('low_x.pickle', 'wb')
        pickle.dump(low_x, low_x_file,-1)
    
    ##################################### Outlier Detection
    '''
    Detect outliers with isolationforest and oneclasssvm.
    See the distribution of outliers based on low dimensional data transformed by manifold learning.
    See the relationship between outliers and churn customer.
    '''
    
    isoForest = IsolationForest(n_jobs=3, verbose = 1)
    isoForest.fit(x_train)
    x_train_outlier_iso = isoForest.predict(x_train)
    plt.figure(figsize = (20,16))
    plt.scatter(low_x[:,0], low_x[:,1], c = x_train_outlier_iso) 
    plt.savefig('t-sne and outliers predicted by isolation forest.png')
    
    if not os.path.exists('IsolationForest.pickle'):
        iso_file = open('IsolationForest.pickle', 'wb')
        pickle.dump(isoForest, iso_file,-1)    
        
        
    OCSVM = OneClassSVM()
    OCSVM.fit(x_train)
    x_train_outlier_svm = OCSVM.predict(x_train)
    plt.figure(figsize = (20,16))
    plt.scatter(low_x[:,0], low_x[:,1], c = x_train_outlier_svm) 
    
    
    
    
    