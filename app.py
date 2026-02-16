import streamlit as st
import pandas as pd
from utils import load_data, preprocess_data
from model import train_model, evaluate_model, predict_price
import plotly.express as px
import datetime
import joblib
import io

st.set_page_config(page_title="Diamond Price Predictor",
                   layout="wide",
                   page_icon="💎")

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load Data
df = load_data()

# Header
st.markdown("<div class='main-title'>💎 Diamond Price Prediction Dashboard</div>", unsafe_allow_html=True)

# Greeting
hour = datetime.datetime.now().hour
greeting = "Good Morning ☀️" if hour < 12 else "Good Afternoon 🌤️" if hour < 17 else "Good Evening 🌙"
st.markdown(f"<div class='greeting'>{greeting}</div>", unsafe_allow_html=True)

# Sidebar
#st.sidebar.title("⚙️ Settings")

section = st.sidebar.radio("Navigate", ["📊 EDA", "🤖 Model Training", "💰 Prediction"])

# ---------------- EDA ----------------
if section == "📊 EDA":

    st.subheader("📊 Exploratory Data Analysis")

    # ---------- Sidebar Filters ----------
    st.sidebar.markdown("### 🔎 Filter Data")

    selected_cut = st.sidebar.multiselect(
        "Select Cut",
        options=df["cut"].unique(),
        default=df["cut"].unique()
    )

    selected_color = st.sidebar.multiselect(
        "Select Color",
        options=df["color"].unique(),
        default=df["color"].unique()
    )

    selected_clarity = st.sidebar.multiselect(
        "Select Clarity",
        options=df["clarity"].unique(),
        default=df["clarity"].unique()
    )

    filtered_df = df[
        (df["cut"].isin(selected_cut)) &
        (df["color"].isin(selected_color)) &
        (df["clarity"].isin(selected_clarity))
    ]

    st.markdown("### 📌 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    st.markdown("---")

    # ---------- KPI Section ----------
    st.markdown("### 📈 Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Diamonds", len(filtered_df))
    col2.metric("Average Price", f"${round(filtered_df['price'].mean(),2)}")
    col3.metric("Average Carat", round(filtered_df['carat'].mean(),2))
    col4.metric("Max Price", f"${filtered_df['price'].max()}")

    st.markdown("---")

    # ---------- Distribution Control ----------
    st.markdown("### 📊 Distribution Analysis")

    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    selected_numeric = st.selectbox("Select Numerical Feature", numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            filtered_df,
            x=selected_numeric,
            marginal="box",
            nbins=40,
            title=f"Histogram of {selected_numeric}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(
            filtered_df,
            x="carat",
            y="price",
            color="cut",
            size="depth",
            hover_data=["color", "clarity"],
            title="Carat vs Price"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ---------- Category Comparison ----------
    st.markdown("### 🏷 Category Comparison")

    cat_col = st.selectbox("Select Category Feature",
                           ["cut", "color", "clarity"])

    fig_box = px.box(
        filtered_df,
        x=cat_col,
        y="price",
        color=cat_col,
        title=f"Price Distribution by {cat_col}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_violin = px.violin(
        filtered_df,
        x=cat_col,
        y="price",
        color=cat_col,
        box=True,
        points="all",
        title=f"Violin Plot - {cat_col} vs Price"
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")

    # ---------- Correlation Heatmap ----------
    st.markdown("### 🔥 Correlation Heatmap")

    corr = filtered_df.corr(numeric_only=True)

    fig_corr = px.imshow(
        corr,
        width=800,
        height=800,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ---------- 3D Visualization ----------
    st.markdown("### 🌐 3D Interactive Visualization")

    fig_3d = px.scatter_3d(
        filtered_df,
        x="carat",
        y="depth",
        z="price",
        color="cut",
        size="table",
        opacity=0.7
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# ---------------- Model Training ----------------
elif section == "🤖 Model Training":
    st.subheader("Train Machine Learning Model")
    model_name = st.selectbox("Select Model",
                              ["LinearRegression",
                               "RandomForest",
                               "GradientBoosting",
                               "DecisionTree"])

    model, X_test, y_test = train_model(df, model_name)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Evaluate Model 📊"):
            scores = evaluate_model(model, X_test, y_test)
            st.success("Model Evaluation Completed")
            st.dataframe(scores)

    with col2:
        # Convert model to bytes
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        st.download_button(
            label="Download Trained Model 💾",
            data=buffer,
            file_name=f"{model_name}_diamond_model.pkl",
            mime="application/octet-stream"
        )

        #if st.download_button(
        #        label="Download Trained Model 💾",
        #        data=buffer,
        ##        file_name=f"{model_name}_diamond_model.pkl",
         #       mime="application/octet-stream"
        #):
        #    st.success("Model Downloaded Successfully ✅")

# ---------------- Prediction ----------------
elif section == "💰 Prediction":

    st.subheader("Base on Random Forest")

    uploaded_model = st.file_uploader("Upload Model (.pkl file)", type=["pkl"])

    #if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    #model = joblib.load("RandomForest_diamond_model.pkl")

    st.subheader("Enter Diamond Details")

    col1, col2 = st.columns(2)

    with col1:
            carat = st.number_input("Carat", 0.0)
            depth = st.number_input("Depth", 0.0)
            table = st.number_input("Table", 0.0)
            x = st.number_input("Length (x)", 0.0)

    with col2:
            y = st.number_input("Width (y)", 0.0)
            z = st.number_input("Height (z)", 0.0)
            cut = st.selectbox("Cut", ['Ideal','Premium','Very Good','Good','Fair'])
            color = st.selectbox("Color", ['D','E','F','G','H','I','J'])
            clarity = st.selectbox("Clarity", ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])

    if st.button("Predict Price 💎"):

        prediction = predict_price(
                model, carat, cut, color,
                clarity, depth, table, x, y, z
            )

        st.markdown(
                f"<div class='prediction-box'>Estimated Price: ${prediction}</div>",
                unsafe_allow_html=True
            )

    else:
        st.info("Please upload a trained model first.")

