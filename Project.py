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


@st.cache(allow_output_mutation=True)
def get_data():
    df = pd.read_csv("diamonds.csv")
    df = df.drop(['Unnamed: 0'], axis=1)
    return df

Data = get_data()

## Title
st.markdown("<h1 style='text-align: center; color: red;'>REPORT 📝</h1>", unsafe_allow_html=True)


## Greeting
now = datetime.datetime.now()
hour = now.hour
if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"
st.write("{}!".format(greeting))

st.markdown("<h4 style='text-align: center;'>========================================================================================</h4>", unsafe_allow_html=True)

clarity_oder = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
code1_s = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5]


##Part -1 EDA
st.sidebar.write('EDA Report')
Dataset = st.sidebar.checkbox("Dataset")
if Dataset:
    st.markdown("<h2 style='text-align: center;'>Dataset Head </h2>", unsafe_allow_html=True)
    st.table(Data.head())
    st.write(Data.shape)

EDA = st.sidebar.checkbox("EDA")
if EDA:
    st.markdown("<h2 style='text-align: center;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    Describe = st.checkbox("Describe")
    Indivual = st.checkbox("Indivual")
    if Describe:
        st.markdown("<h3 style='text-align: center;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
        st.table(Data.describe())
        df = Data.corr()
        df = round(df,2)
        fig = px.imshow(df, width=800, height=800, text_auto=True, color_continuous_scale='RdBu_r')
        st.markdown("<h3 style='text-align: center;'>Correlation</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig)



    if Indivual:
        col1, col2 = st.columns([1, 1])
        select_column = col1.selectbox("Select Column", ("color", "clarity", "cut", "carat", "depth", "table","x","y","z"))
        cart = ["color", "clarity", "cut"]
        cl1, cl2 = st.columns([1, 1])
        if select_column in cart:
            df1 = Data.groupby(by=[select_column]).size().reset_index(name="counts")
            df1['ss'] = df1[select_column].replace(clarity_oder, code1_s)
            df1 = pd.DataFrame(df1.sort_values(by=['ss']))
            df1 = df1.drop('ss', axis=1, inplace=False)
            df1 = df1.reset_index(drop=True)
            Data['PPC'] = Data['price']/Data['carat']
            fig = px.bar(df1, x=select_column, y="counts", title="Count Plot", width=400, height=400)
            fig1 = px.pie(df1, values='counts', names=select_column, title='Pie Chart', width=400,
                      height=400)
            fig0 = px.box(Data, y="price", x=select_column, width=800, height=600)
        else:
            x = select_column
            fig = px.box(Data, y=x, title="Box Plot", width=400, height=400)
            fig1 = px.histogram(Data, x=x, title="Histogram", width=400, height=400)
            fig0 = ff.create_2d_density(Data['price'], Data[x], width=800, height=600)
            #iplot(fig)
        cl1.plotly_chart(fig)
        cl2.plotly_chart(fig1)
        st.plotly_chart(fig0)

        # fig = px.violin(Data, y="price", x="color", width=800, height=500)
        # st.plotly_chart(fig)
        # fig = plt.figure(figsize=(10, 4))
        # sns.violinplot(data=Data, x="color", y="price", split=True)
        # st.pyplot(fig)

##Model
Method = st.sidebar.selectbox("Select Method",("Regression","Classification","NLP"))
le = LabelEncoder()

Data['cut'] = le.fit_transform(Data['cut'])
Data['color'] = le.fit_transform(Data['color'])
Data['clarity'] = le.fit_transform(Data['clarity'])

X=Data.drop('price',axis=1)
y=Data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Re = st.sidebar.selectbox("Select",("LinearSVR","KNeighborsRegressor","LinearRegression","RandomForestRegressor",
                                    "GradientBoostingRegressor","DecisionTreeRegressor"))

if Re == "LinearSVR":
    pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearSVR(random_state = 42))])

    model = pipe1.fit(X_train, y_train)

    y_pred = model.predict(X_test)

elif Re == "KNeighborsRegressor":
    pipe2 = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', KNeighborsRegressor())])

    model = pipe2.fit(X_train, y_train)

    y_pred = model.predict(X_test)

elif Re == "LinearRegression":
    pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())])

    model = pipe3.fit(X_train, y_train)

    y_pred = model.predict(X_test)

elif Re == "RandomForestRegressor":
    model = RandomForestRegressor(random_state = 42).fit(X_train, y_train)

    y_pred = model.predict(X_test)

elif Re == "GradientBoostingRegressor":
    model = GradientBoostingRegressor(random_state = 42).fit(X_train, y_train)

    y_pred = model.predict(X_test)

elif Re == "DecisionTreeRegressor":
    model = DecisionTreeRegressor(random_state = 42).fit(X_train, y_train)

    y_pred = model.predict(X_test)

result = st.sidebar.checkbox("Show Result Score")
a = mean_absolute_error(y_test, y_pred)
b = mean_absolute_percentage_error(y_test, y_pred)
c = mean_squared_error(y_test, y_pred, squared = True)
d = mean_squared_error(y_test, y_pred, squared = False)
e = r2_score(y_test, y_pred)
Table = {
  "Results": ["MAE", "MAPE", "MSE", "RMSE", "R2"],
  "Score": [a, b, c, d, e]
}
table = pd.DataFrame(Table)
if result:
    st.title(Re)
    st.table(table)
