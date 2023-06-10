import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler


def prepare():
    df = pd.read_csv("klse.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df.drop("2023-04-20", axis=0, inplace=True)

    df.drop("Dividends", axis=1, inplace=True)
    df.drop("Stock Splits", axis=1, inplace=True)

    df["Next Close"] = df["Close"].shift(-1)
    df.loc["2023-04-19", "Next Close"] = 1422.10998535156

    train_test_point = int(len(df) * 0.8)

    train_df = df.iloc[:train_test_point]
    test_df = df.iloc[train_test_point:]

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(train_df.iloc[:, [0, 1, 2, 4]])
    y_train_scaled = y_scaler.fit_transform(train_df.iloc[:, [5]])

    X_test_scaled = X_scaler.transform(test_df.iloc[:, [0, 1, 2, 4]])
    y_test_scaled = y_scaler.transform(test_df.iloc[:, [5]])

    return (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        X_scaler,
        y_scaler,
    )


def prepare_lstm(X, y):
    X_60d = []
    y_shifted = []

    for i in range(60, len(X)):
        X_60d.append(X[i - 60 : i, :])
        y_shifted.append(y[i])

    return np.array(X_60d), np.array(y_shifted)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)

        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)

        return K.sum(x * alpha, axis=1)
