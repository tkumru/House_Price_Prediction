from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from lazypredict.Supervised import LazyRegressor

class LinearRegressionSklearn:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        
    def initialize_df(self):
        self.df = self.df.dropna()
        
        self.df = pd.get_dummies(self.df, drop_first=True)
        
    def model(self):
        X, y = self.df.drop(self.target_col, axis=1), self.df[[self.target_col]]

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, 
                                                                      test_size=0.2, 
                                                                      random_state=7)
        
        self.regression_model = LinearRegression().fit(X_train, y_train)
        
    def get_score(self):
        return self.regression_model.score(self.X_test, self.y_test)
