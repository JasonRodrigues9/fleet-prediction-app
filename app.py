
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import plotly.express as px

st.set_page_config(page_title="Fleet Future Order Prediction", layout="wide")

st.title("🚛 Fleet Future Order Prediction App")
st.write("Upload your historical booking Excel file to predict future equipment demand.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    required_cols = [
        "Date",
        "Priority",
        "Equipment type",
        "Model",
        "Qty"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:

        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

        encoders = {}

        categorical_cols = ["Priority", "Equipment type", "Model"]

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        X = df[["Priority", "Equipment type", "Model", "Month", "Week", "Year"]]
        y = df["Qty"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        st.subheader("📊 Model Accuracy")
        st.metric("Mean Absolute Error", round(mae, 2))

        st.subheader("🔮 Predict Future Orders")

        future_month = st.selectbox("Select Future Month", list(range(1,13)))

        models = encoders["Model"].classes_

        future_predictions = []

        for m in models:

            model_encoded = encoders["Model"].transform([m])[0]

            avg_priority = int(df["Priority"].mean())
            avg_equipment = int(df["Equipment type"].mean())

            pred_input = pd.DataFrame({
                "Priority": [avg_priority],
                "Equipment type": [avg_equipment],
                "Model": [model_encoded],
                "Month": [future_month],
                "Week": [1],
                "Year": [2026]
            })

            pred_qty = model.predict(pred_input)[0]

            future_predictions.append({
                "Model": m,
                "Predicted Future Orders": round(max(pred_qty,0),2)
            })

        pred_df = pd.DataFrame(future_predictions)

        st.dataframe(pred_df)

        fig = px.bar(
            pred_df,
            x="Model",
            y="Predicted Future Orders",
            title="Predicted Future Orders by Model"
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction CSV",
            csv,
            "future_order_predictions.csv",
            "text/csv"
        )
