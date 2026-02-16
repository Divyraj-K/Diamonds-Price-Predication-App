import pandas as pd

cut_mapping = {'Ideal':2, 'Premium':1, 'Very Good':3, 'Good':4, 'Fair':5}
color_mapping = dict(zip(['D','E','F','G','H','I','J'], range(7)))
clarity_mapping = dict(zip(['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'], range(8)))

def load_data():
    df = pd.read_csv("diamonds.csv")
    df = df.drop(['Unnamed: 0'], axis=1)
    return df

def preprocess_data(df):
    df['cut'] = df['cut'].map(cut_mapping)
    df['color'] = df['color'].map(color_mapping)
    df['clarity'] = df['clarity'].map(clarity_mapping)
    return df
