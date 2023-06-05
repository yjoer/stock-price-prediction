import datetime
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.palettes import Category10
from bokeh.plotting import figure
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tensorflow import keras


@st.cache_data
def get_data():
    df = pd.read_csv("klse.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df.drop("2023-04-20", axis=0, inplace=True)

    df.drop("Dividends", axis=1, inplace=True)
    df.drop("Stock Splits", axis=1, inplace=True)

    return df


@st.cache_data
def load_scaler():
    X_scaler = pickle.load(open("artifacts/X_scaler.pkl", "rb"))
    y_scaler = pickle.load(open("artifacts/y_scaler.pkl", "rb"))

    return X_scaler, y_scaler


@st.cache_data
def prepare_lstm(X, y, selected_range):
    X_60d = []
    y_shifted = []

    for i in selected_range:
        X_60d.append(X[i - 60 : i, :])
        y_shifted.append(y[i])

    return np.array(X_60d), np.array(y_shifted)


@st.cache_data
def load_lstm_model():
    lstm_256 = keras.models.load_model("artifacts/best_lstm_256.h5")

    return lstm_256


def reg_scores(y_test, y_pred):
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


df = get_data()

X_scaler, y_scaler = load_scaler()

X = X_scaler.transform(df.iloc[:, [0, 1, 2, 4]])
y = y_scaler.transform(df.iloc[:, [3]])

lstm_256 = load_lstm_model()

st.markdown("## Same-Day Predictions")

date_range = st.date_input(
    label="Date Range",
    value=(datetime.date(2023, 1, 1), datetime.date(2023, 4, 19)),
)

date_range = pd.to_datetime(date_range)
date_range = date_range.tz_localize(tz="Asia/Kuala_Lumpur")

if len(date_range) < 2:
    st.stop()


date_range_mask = (df.index > date_range[0]) & (df.index < date_range[1])
df_selected = df.loc[date_range_mask]
index_values = [df.index.get_loc(date) for date in df_selected.index]

if len(index_values) == 0:
    st.stop()

if len(index_values) > 0 and index_values[0] - 60 < 0:
    st.stop()

st.markdown("### Predictions")

X_test_scaled_60d, y_test_scaled_shifted = prepare_lstm(X, y, index_values)
y_pred = y_scaler.inverse_transform(lstm_256.predict(X_test_scaled_60d))
df_selected["Predicted Close"] = y_pred
df_selected["Deviation"] = df_selected["Predicted Close"] - df_selected["Close"]

p = figure(x_axis_type="datetime")

p.line(
    df_selected.index,
    df_selected["Close"],
    legend_label="Close",
    color=Category10[3][0],
    line_width=2,
)
p.line(
    df_selected.index,
    df_selected["Predicted Close"],
    legend_label="Predicted Close",
    color=Category10[3][1],
    line_width=2,
)

st.bokeh_chart(p, use_container_width=True)
st.dataframe(df_selected)

col1, col2 = st.columns([1, 3])

col1.markdown("### Errors")

scores = reg_scores(y_pred, y_scaler.inverse_transform(y_test_scaled_shifted))
scores = pd.DataFrame(scores, index=["Scores"]).T
col1.table(scores)

col2.markdown("### Error Distribution")

col2_1, col2_2 = col2.columns([1, 2])

percentiles = {
    "90th Percentile": df_selected["Deviation"].quantile(0.9),
    "95th Percentile": df_selected["Deviation"].quantile(0.95),
    "99th Percentile": df_selected["Deviation"].quantile(0.99),
}

col2_1.table(df_selected["Deviation"].describe())
col2_2.table(percentiles)
