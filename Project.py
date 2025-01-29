import pandas as pd
import numpy as np
import streamlit as st
import datetime
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


st.set_page_config(layout="wide")


#@st.cache(allow_output_mutation=True)
def get_data():
    df = pd.read_csv("diamonds.csv")
    df = df.drop(['Unnamed: 0'], axis=1)
    return df

Data = get_data()

st.write(Data)
