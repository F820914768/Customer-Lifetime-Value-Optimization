# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:07:34 2019

@author: 82091
"""
from SaveLoad import save_pickle, list_all_pickle, load_pickle
import tensorflow as tf

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

class AutoEncoder:
    def __init__(self, dimension = 3, input_shape = (39,), 
                 num_layers = 0, **kwargs):
        
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.strategy = tf.distribute.MirroredStrategy(**kwargs)
        self.dimension = dimension
        
    def create_model(self):
        with self.strategy.scope():
            
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Dense(64, activation='relu', input_shape = self.input_shape))
            for i in range(self.num_layers):
                self.model.add(tf.keras.layers.Dense(64, activation='relu'))
            self.model.add(tf.keras.layers.Dense(self.dimension))
            self.model.add(tf.keras.layers.Dense(64, activation='relu'))
            for i in range(self.num_layers):
                self.model.add(tf.keras.layers.Dense(64, activation='relu'))
            self.model.add(tf.keras.layers.Dense(self.input_shape[0], activation='sigmoid'))
            
            self.model.compile(loss='MSE',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['MSE'])
        import os
        checkpoint_dir = 'training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        self.callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir='encoder-logs'),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                save_weights_only=True),
                tf.keras.callbacks.LearningRateScheduler(decay)
                ]
            
    def create_datasets(self, x_train, x_test, BATCH_SIZE_PER_REPLICA = 32):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
        self.BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
        self.train_dataset = train_dataset.batch(self.BATCH_SIZE)
        self.eval_dataset = test_dataset.batch(self.BATCH_SIZE)
        
    def fit(self, epochs = 4):        
        self.model.fit(self.train_dataset, epochs=epochs, callbacks=self.callbacks)
        eval_loss, eval_acc = self.model.evaluate(self.eval_dataset)
        print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


if __name__ == '__main__':
    pickle_list = list_all_pickle()
    #x_train, x_test, y_train, y_test = load_pickle('dataWithtype.pickle')
    #x_train = x_train[:,4:]
    #x_test = x_test[:,4:]
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np
    x = np.array(pd.read_excel('all_data_after_basic_transformation_except_y.xlsx',index_col=0))
    x = np.delete(x,[1,2,3], axis = 1)
    citizen = np.unique(x[:,0])
    x[x[:,0]==citizen[0],0] = 0
    x[x[:,0]==citizen[1],0] = 1
    y = x
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
    model = AutoEncoder(num_layers = 1,
                        dimension = 2, input_shape = (35,))
    model.create_datasets(x_train, x_test)
    model.create_model()
    model.fit(epochs=4)
    
    model.model.save('autoencoder1.h5')
    encoder = model.model
    
    print(encoder.predict(x_test[:2,:]))
    print(x_test[:2,:])

    