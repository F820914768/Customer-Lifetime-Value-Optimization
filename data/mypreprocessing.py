# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:43:15 2019

@author: 82091
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class data_selector(BaseEstimator, TransformerMixin):
    '''
    The objective of this transformer is to split data into numerical and categorical for building pipeline.
    select either numerical data or categorical data.
    When num_feature == True, select numerical data; when num_feature==False, select categorical data
    '''
    def __init__(self, attr,num_feature = True):
        self.num_feature = num_feature     # defining which type of feature to select
        self.attr = attr                    # dictionary to split features
    def fit(self, X, y=None):
        return self                        # doing nothing at fitting
    def transform(self, X, y=None):
        if self.num_feature:                # if num_feature is True, select numerical feature
            return X[self.attr['num']]
        else:                               # if num_feature is False, select categorical feature
            return X[self.attr['cat']]
        
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

class MyFeatureDeletor(BaseEstimator, TransformerMixin):
    '''
    This is for deleting duplicate features.
    '''
    def __init__(self, delete_cols):
        self.delete_cols = delete_cols
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        return np.delete(X, self.delete_cols, axis = 1)
    
class CustomerClassificationModel:
    def __init__(self, model_path):
        self.model_path = model_path
    def fit(self, X, y=None):
        import pickle
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        return self
    def transform(self, X, y=None):
        customer_type_feature = self.model.predict(X).reshape((-1,1))
        customer_type_feature = OneHotEncoder().fit_transform(customer_type_feature).toarray()
        return np.c_[X, customer_type_feature]

def create_preprocessing_pipeline(attr, duplicate_indices, model_path):
    '''
    1. split feature into categorical and numerical
    2. normalize numerical features
    3. encode categorical features with one hot encoder
    4. combine two types of features
    5. delete duplicate features which have correlation of 1
    6. add customer type feature predicted by logistic regression
    '''
    num_feature_transformer = Pipeline([('num_selector', data_selector(attr)),
                                        ('normalization', StandardScaler())])
    cat_feature_transformer = Pipeline([('cat_selector',data_selector(attr,num_feature=False)),
                                        ('label_encoder',MyLabelEncoder()),
                                        ('one_hot_encoder', OneHotEncoder(sparse=False))])
    feature_union = FeatureUnion([('num_feature',num_feature_transformer),
                                  ('cat_feature',cat_feature_transformer)])
    full_pipeline = Pipeline([('feature_union', feature_union),
                              ('feature_deltion', MyFeatureDeletor(duplicate_indices)),
                              ('customer_type_adder', CustomerClassificationModel(model_path))])
    
    return full_pipeline

def data_cleaning(attribute, df):
    df['TotalCharges'] = df['TotalCharges'].str.strip()
    for i in range(len(df)):
        try:
            float(df.loc[i,:]['TotalCharges'])
        except ValueError:
            df.loc[i]['TotalCharges'] = np.nan
    df.drop(df[df['TotalCharges']==''].index, inplace = True)
    df['TotalCharges'] = df['TotalCharges'].astype('float')
    return df