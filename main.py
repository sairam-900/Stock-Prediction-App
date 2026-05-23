import streamlit as st
from datetime import date

import pandas as pd
try:
    import yfinance as yf  # type: ignore
except Exception:
    # If yfinance isn't available, show an error in the Streamlit app and stop.
    import streamlit as _st
    _st.error("Missing dependency: yfinance. Install it with `pip install yfinance`.")
    _st.stop()

try:
    from prophet import Prophet  # type: ignore
    from prophet.plot import plot_plotly  # type: ignore
except Exception:
    import streamlit as _st
    _st.error("Missing dependency: prophet. Install it with `pip install prophet`.")
    _st.stop()

try:
    from plotly import graph_objs as go  # type: ignore
except Exception:
    import streamlit as _st
    _st.error("Missing dependency: plotly. Install it with `pip install plotly`.")
    _st.stop()


# ---------------- APP SETTINGS ----------------

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Prediction App")


# ---------------- STOCK SELECTION ----------------

stocks = ("AAPL", "GOOG", "MSFT", "GME")

selected_stock = st.selectbox(
    "Select dataset for prediction",
    stocks
)

n_years = st.slider(
    "Years of prediction:",
    1,
    4
)

period = n_years * 365


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data(ticker):

    data = yf.download(
        ticker,
        START,
        TODAY
    )

    data.reset_index(inplace=True)

    return data


data_load_state = st.text("Loading data...")

data = load_data(selected_stock)

data_load_state.text("Loading data...done!")


# ---------------- RAW DATA ----------------

st.subheader("Raw Data")

st.write(data.tail())


# ---------------- PLOT RAW DATA ----------------

def plot_raw_data():

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Open'],
            name='Stock Open'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            name='Stock Close'
        )
    )

    fig.update_layout(
        title="Time Series Data",
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)


plot_raw_data()


# ---------------- FORECASTING ----------------

# Create dataframe
df_train = pd.DataFrame()

# Add date column
df_train["ds"] = data["Date"]

# Add close price column properly
df_train["y"] = data["Close"].values

# Convert to numeric
df_train["y"] = pd.to_numeric(
    df_train["y"],
    errors="coerce"
)

# Remove missing values
df_train.dropna(inplace=True)


# ---------------- PROPHET MODEL ----------------

model = Prophet()

model.fit(df_train)


# ---------------- FUTURE DATAFRAME ----------------

future = model.make_future_dataframe(
    periods=period
)


# ---------------- PREDICTION ----------------

forecast = model.predict(future)


# ---------------- FORECAST DATA ----------------

st.subheader("Forecast Data")

st.write(forecast.tail())


# ---------------- FORECAST PLOT ----------------

st.subheader("Forecast Plot")

fig1 = plot_plotly(
    model,
    forecast
)

st.plotly_chart(fig1)


# ---------------- FORECAST COMPONENTS ----------------

st.subheader("Forecast Components")

fig2 = model.plot_components(forecast)

st.write(fig2)