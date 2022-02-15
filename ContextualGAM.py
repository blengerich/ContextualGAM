import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
import interpret
import sys
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import f1_score
import os

from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from matplotlib.patches import Rectangle

class NGAM(tf.keras.layers.Layer):
    def __init__(self, input_width, output_width, depth, width, activation='swish',
                final_activation='linear', boolean_feats=None):
        super(NGAM, self).__init__()
        self.models = []
        self.input_width = input_width
        self.output_width = output_width
        for i in range(input_width):
            my_layers = [tf.keras.layers.Flatten()]
            if boolean_feats is None or boolean_feats[i] == False: # if boolean feature, don't need hidden layers
                my_layers.extend([
                    tf.keras.layers.Dense(
                        width, activation=activation,
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01))
                    for _ in range(depth)])
            my_layers.append(
                tf.keras.layers.Dense(
                    output_width, activation=final_activation,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)))
            self.models.append(tf.keras.models.Sequential(my_layers))
        self._name = "NGAM"
            
    def build(self, input_shapes):
        pass
    
    def call(self, my_input):
        outputs = [self.models[i](
            tf.keras.layers.Lambda(
                lambda x: x[:, i],
                output_shape=(None, self.output_width))(my_input))
                   for i in range(self.input_width)]
        return outputs

class Explainer(tf.keras.layers.Layer):
    def __init__(self, archetype_init, activity_regularizer, **kwargs):
        super(Explainer, self).__init__(kwargs)
        self.k, self.d = archetype_init.shape
        self.archetypes = self.add_weight("archetypes", shape=archetype_init.shape,
                                          initializer=tf.constant_initializer(archetype_init),
                                          trainable=True)
        self.activity_regularizer = activity_regularizer
        
    def build(self, input_shapes):
        pass

    def call(self, subtype): # Subtype is of shape (None x k)
        return tf.tensordot(subtype, self.archetypes, axes=1)

class BatchDot(tf.keras.layers.Layer):
    def __init__(self):
        super(BatchDot, self).__init__()

    def build(self, input_shapes):
        pass

    def call(self, A, B):
        return tf.keras.backend.batch_dot(A, B)
    
class ContextualGAM:
    def __init__(self, encoder_input_shape, encoder_output_shape, encoder_depth, 
                 dict_shape, X_shape, sample_specific_loss_params, archetype_loss_params,
                 archetype_init, learning_rate=1e-3, encoder_width=32, 
                 skip_encoder_depth=1, skip_encoder_width=4, skip_activation='linear',
                 tf_dtype=tf.dtypes.float32, 
                 contextual_bools=None, activation='linear', encoder_final_activation='linear',
                 use_skip=False, base_model=None):
        super(ContextualGAM, self).__init__()
        self.C = tf.keras.layers.Input(shape=encoder_input_shape, dtype=tf_dtype, name="C")
        self.X = tf.keras.layers.Input(shape=X_shape, dtype=tf_dtype, name="X")
        self.base_y = tf.keras.layers.Input(shape=(1, ), dtype=tf_dtype, name='base_y')
        self.base_model = base_model
        
        # Encoder: context --> subtype
        self.C_flat = tf.keras.layers.Flatten()(self.C)
        self.encoder_gam = tf.reduce_sum(
            NGAM(encoder_input_shape, encoder_output_shape[0],
                 depth=encoder_depth, width=encoder_width,
                 boolean_feats=contextual_bools,
                 activation=activation, 
                 final_activation=encoder_final_activation)(self.C_flat), axis=0)
        self.encoder = tf.keras.models.Model(inputs=self.C, outputs=self.encoder_gam)
        self.encodings = self.encoder(self.C)
        self.explainer = Explainer(
            archetype_init,
            activity_regularizer=tf.keras.regularizers.l1(sample_specific_loss_params['l1']))
        self.sample_models = self.explainer(self.encodings)
        self.explainable_risk = BatchDot()(self.X, self.sample_models)
        self.total_risk = self.base_y + self.explainable_risk
        if use_skip:
            self.skip_encoder = tf.reduce_sum(
                NGAM(encoder_input_shape, 1,
                     depth=skip_encoder_depth,
                     width=skip_encoder_width,
                     activation=skip_activation)(self.C_flat),
                axis=0)
            self.skip_encoder_model = tf.keras.models.Model(
                inputs=self.C, outputs=self.skip_encoder)
            self.skip_encodings = self.skip_encoder_model(self.C)
            self.total_risk += self.skip_encodings
        self.outputs = tf.nn.sigmoid(self.total_risk)
        self.model = tf.keras.models.Model(inputs=(self.C, self.X, self.base_y),
                                           outputs=self.outputs)

        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = ['AUC']

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.model.compile(
            loss=self.loss,
            optimizer=self.opt, 
            metrics=self.metrics
        )
    
    def get_embeddings(self, C):
        return self.encoder(C)
    
    def get_sample_models(self, C):
        return self.explainer(self.get_embeddings(C))
    
    def predict_proba(self, C, X, C_base=None):
        if self.base_model is not None:
            if C_base is None:
                base_y = self.base_model.predict_proba(C)[:, 1].astype(np.float32)
            else:
                base_y = self.base_model.predict_proba(C_base)[:, 1].astype(np.float32)
        else:
            base_y = np.zeros((len(C), 1))
        return self.model.predict({"C": C, "X": X, "base_y": base_y})
    
    def predict(self, C, X, C_base=None):
        return np.round(self.predict_proba(C, X, C_base))
    
    def fit(self, C_train, X_train, Y_train, C_base=None,
            max_epochs=500, verbose=1, batch_size=16,
            early_stopping_epochs=10, val_split=0.2):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=early_stopping_epochs,
            restore_best_weights=True, mode='max', min_delta=1e-4)
        if self.base_model is not None:
            if C_base is None:
                base_y = self.base_model.predict_proba(C_train)[:, 1]
            else:
                base_y = self.base_model.predict_proba(C_base)[:, 1].astype(np.float32)
        else:
            base_y = np.zeros((len(X_train), 1))

        history = self.model.fit(
            {"X": X_train.astype(np.float32),
             "C": C_train.astype(np.float32),
             "base_y": base_y.astype(np.float32)},
            y=Y_train, epochs=max_epochs, callbacks=[early_stopping],
            validation_split=val_split, verbose=verbose, batch_size=batch_size)
        epoch = np.argmax(history.history['val_auc'])
        val_auc = history.history['val_auc'][epoch]
        train_auc = history.history['auc'][epoch]
        return epoch, train_auc, val_auc