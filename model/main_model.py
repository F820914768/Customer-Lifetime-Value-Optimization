# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 03:30:19 2019

@author: 82091
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from model.mypreprocessing import create_preprocessing_pipeline, data_cleaning
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from SaveLoad import save_pickle


file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
logicmodel_path = 'Logistic-Model-for-customer-classification.pickle'
attr, attr_indicator, duplicate_features = pickle.load(open('attr_split-attr_afterdummy_indicator-duplicate_feature.pickle','rb'))
feature_indices = attr_indicator[duplicate_features]

preproce_pipeline = create_preprocessing_pipeline(attr, feature_indices, logicmodel_path)
df = pd.read_csv(file_name)
df = data_cleaning('TotalCharges',df)

y = df['Churn']
df.drop('Churn', axis = 1, inplace = True)

x_train, x_test, y_train, y_test = train_test_split(df,y,test_size = 0.3,
                                                        random_state=12)
id_train, id_test = x_train['customerID'], x_test['customerID']
x_train.drop('customerID',axis=1,inplace=True)
x_test.drop('customerID',axis=1,inplace=True)

labelencoder_y = LabelEncoder()
x_train = preproce_pipeline.fit_transform(x_train)
x_test = preproce_pipeline.fit_transform(x_test)
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.fit_transform(y_test)

save_pickle('dataWithtype.pickle',[x_train, x_test, y_train, y_test])

svm = SVC()
svm.fit(x_train, y_train)
print(accuracy_score(y_test, svm.predict(x_test)))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


####################################################################

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE_PER_REPLICA = 32
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = train_dataset.batch(BATCH_SIZE)
eval_dataset = test_dataset.batch(BATCH_SIZE)


#
#tf.config.gpu.set_per_process_memory_fraction(0.4)
with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape = (43,)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
  
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
    
model.fit(train_dataset, epochs=6, callbacks=callbacks)

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))




