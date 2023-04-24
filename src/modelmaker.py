import pandas as pd
import numpy as np
from scikeras.wrappers import KerasClassifier

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Concatenate, Activation, Dot, Flatten, Dropout, Attention)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MakeModel():

    def __init__(self, df, cols):
        df = df.reset_index(drop=True)
        features = [col for col in df.columns if col not in ['id', 'date', 'time', 'label']]
        colTransformer = ColumnTransformer([('num', StandardScaler(), features)], remainder='passthrough')
        df_values = colTransformer.fit_transform(df)
        df = pd.DataFrame(data=df_values, columns=df.columns)

        self.number_of_classes = df['label'].unique().shape[0]
        if (self.number_of_classes == 2.0):
            df['label'] = df['label'].replace(2.0, 1.0)

        X, y = df[cols].values, to_categorical(df['label'], self.number_of_classes)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        window_size = 1
        X = X.reshape((X.shape[0], window_size, X.shape[1] // window_size))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = self.create_model()

    def fit_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=80,
                       batch_size=64,
                       validation_data=(self.X_test, self.y_test),
                       callbacks=[early_stopping],
                       verbose=2)

    def get_model(self):
        return self.model

    def get_train_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_model(self):
        input_shape = self.X_train.shape[1:]
        input_layer = Input(shape=input_shape)
        lstm0 = LSTM(64, return_sequences=True)(input_layer)
        lstm1 = LSTM(64, return_sequences=True)(lstm0)
        dropout0 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        dropout1 = Dropout(0.2)(lstm2)
        attention = Attention()([lstm2, lstm2])
        flatten = Flatten()(attention)
        out = Dense(self.number_of_classes, activation='softmax')(flatten)
        model = Model(inputs=input_layer, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
