from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from utils import preprocess_data
import pandas as pd
import joblib

def train_model(df, model_name):

    df = preprocess_data(df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor()
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2 Score (%)": r2_score(y_test, y_pred) * 100
    }

    return pd.DataFrame(results.items(), columns=["Metric", "Score"])


def predict_price(model, carat, cut, color, clarity, depth, table, x, y, z):

    from utils import cut_mapping, color_mapping, clarity_mapping

    data = pd.DataFrame([{
        'carat': carat,
        'cut': cut_mapping[cut],
        'color': color_mapping[color],
        'clarity': clarity_mapping[clarity],
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }])

    prediction = model.predict(data)[0]
    return round(prediction, 2)
