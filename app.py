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
from statsmodels.tsa.arima.model import ARIMA
from tensorflow import keras


@st.cache_data
def get_data():
    df = pd.read_csv("klse.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index = df.index.to_period("D")
    df.drop("2023-04-20", axis=0, inplace=True)

    df_add = pd.read_csv("klse_june.csv")
    df_add["Date"] = pd.to_datetime(df_add["Date"])
    df_add.set_index("Date", inplace=True)
    df_add.index = df_add.index.to_period("D")

    df = pd.concat([df, df_add])

    start_date = df.index[0].to_timestamp()
    end_date = df.index[-1].to_timestamp()

    new_dates = pd.date_range(start=start_date, end=end_date, freq="D").to_period("D")
    df = df.reindex(new_dates)

    df.drop("Dividends", axis=1, inplace=True)
    df.drop("Stock Splits", axis=1, inplace=True)

    return df


@st.cache_data
def get_intraday_data():
    df = pd.read_csv("klse_2T.csv")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.index = df.index.to_period("2T")

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


def train_arima_daily_model(X_train):
    model = ARIMA(X_train, order=(2, 1, 2))

    return model.fit()


def train_arima_intraday_model(X_train):
    model = ARIMA(X_train, order=(2, 1, 2))

    return model.fit()


def reg_scores(y_test, y_pred):
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


st.set_page_config(initial_sidebar_state="collapsed")

with st.sidebar:
    train_daily = st.button("Train Daily")

    if train_daily:
        df = get_data()
        start_date = pd.to_datetime("2023-04-20").to_period("D")
        X = df.loc[df.index < start_date, ["Close"]]

        model = train_arima_daily_model(X)
        pickle.dump(model, open("artifacts/arima_daily.pkl", "wb"))

    train_intraday = st.button("Train Intraday")

    if train_intraday:
        df = get_intraday_data()

        model = train_arima_intraday_model(X)
        pickle.dump(model, open("artifacts/arima_intraday.pkl", "wb"))

df = get_data()
df_intraday = get_intraday_data()

st.markdown("## Daily Stock Price Predictions")

col1, col2, col3, col4 = st.columns([3, 1, 3, 2])

if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "period"


def date_range_callback():
    st.session_state["input_mode"] = "range"


def date_period_callback():
    st.session_state["input_mode"] = "period"


date_range = col1.date_input(
    label="Select a Date Range",
    on_change=date_range_callback,
    value=(datetime.date(2023, 4, 16), datetime.date(2023, 4, 23)),
)

col2.markdown(
    '<div style="margin: 24px 0; text-align: center;">OR</div>',
    unsafe_allow_html=True,
)

start_date, period = col3.date_input(
    label="Select a Start Date",
    on_change=date_period_callback,
    value=datetime.date(2023, 4, 16),
), col4.number_input(
    label="Period (Days)",
    min_value=1,
    on_change=date_period_callback,
    value=7,
    step=1,
)

if len(date_range) < 2:
    st.stop()

date_range = pd.to_datetime(date_range).to_period("D")
start_date = pd.to_datetime(start_date).to_period("D")

if st.session_state["input_mode"] == "range":
    date_range_mask = (df.index >= date_range[0]) & (df.index <= date_range[1])
elif st.session_state["input_mode"] == "period":
    date_range_mask = (df.index >= start_date) & (df.index <= start_date + period - 1)

df_selected = df.loc[date_range_mask]

arima_daily = pickle.load(open("artifacts/arima_daily.pkl", "rb"))

start_index = 0
end_index = 0

if st.session_state["input_mode"] == "range":
    start_index = (date_range[0] - df.index[0]).n
    end_index = (date_range[1] - df.index[0]).n
elif st.session_state["input_mode"] == "period":
    start_index = (start_date - df.index[0]).n
    end_index = start_index + period - 1

predictions = arima_daily.get_prediction(start=start_index, end=end_index)
y_pred = predictions.predicted_mean

df_selected = df_selected.combine_first(
    pd.DataFrame(y_pred, columns=["predicted_mean"])
)
df_selected.rename(columns={"predicted_mean": "Predicted Close"}, inplace=True)
df_selected["Deviation"] = df_selected["Predicted Close"] - df_selected["Close"]

col1, col2, col3 = st.columns([1, 1, 1])

if st.session_state["input_mode"] == "range":
    start_close = df[df.index <= date_range[0]]["Close"].dropna()[-1]
    end_close_actual = df[df.index <= date_range[1]]["Close"].dropna()[-1]
elif st.session_state["input_mode"] == "period":
    start_close = df[df.index <= start_date]["Close"].dropna()[-1]
    end_close_actual = df[df.index <= start_date + period - 1]["Close"].dropna()[-1]

end_close = df_selected["Predicted Close"][-1]

delta = np.round((end_close - start_close) / start_close * 100, 2)
delta_actual = np.round((end_close_actual - start_close) / start_close * 100, 2)

col1.metric(
    label="Symbol",
    value="^KLSE",
    delta=f"{len(df.dropna())} Days Traded",
    delta_color="off",
)
col2.metric(label="Predicted Close", value=np.round(end_close, 4), delta=f"{delta}%")
col3.metric(
    label="Actual Close",
    value=np.round(end_close_actual, 4),
    delta=f"{delta_actual}%",
)

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

show_ci = st.checkbox(label="Show 95% Confidence Interval")

if show_ci:
    bounds = predictions.conf_int(0.05)
    lower, upper = bounds["lower Close"], bounds["upper Close"]

    reverse_upper = upper[::-1]
    close_range = np.append(lower, reverse_upper)
    dates = np.append(df_selected.index, df_selected.index[::-1])

    p.patch(dates, close_range, color=Category10[3][1], fill_alpha=0.3)

st.bokeh_chart(p, use_container_width=True)
st.dataframe(df_selected)

col1, col2 = st.columns([1, 3])

col1.markdown("### Errors")

scores_df = df_selected[["Close", "Predicted Close"]].dropna()

if len(scores_df) == 0:
    st.text("The ground truth is not available for error calculation.")
    st.stop()

scores = reg_scores(scores_df["Close"], scores_df["Predicted Close"])
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
