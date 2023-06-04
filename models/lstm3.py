import json

from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model_lstm3():
    model = Sequential()

    model.add(LSTM(256, recurrent_dropout=0.2, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, recurrent_dropout=0.2, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(64, recurrent_dropout=0.2))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    opt = Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


def train_model_lstm3(X_train, y_train, X_val, y_val):
    es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    model = build_model_lstm3()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10000,
        batch_size=32,
        callbacks=[es],
    )

    return history


def cv_model_lstm3(X_train, y_train):
    histories = []
    losses = dict()

    tscv = TimeSeriesSplit()

    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        history = train_model_lstm3(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
        histories.append(history)

    for i, history in enumerate(histories):
        losses[f"fold_{i}"] = history.history

    with open("logs/cross_val_lstm3.json", "w") as f:
        f.write(json.dumps(losses, indent=2))
