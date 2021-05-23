import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class MultiPolyModel:
    '''
    Fits multiple polynomial models with different degrees. Prediction value is average value of 
    predicted values from different models (weigths can be used).
    
    If 'max_degree = min_degree' then this is just simple single polynomial model with degree of 
    'max_degree' (or 'max_degree').
    '''
    degrees = None
    weights = None
    models = None
    
    debug = False
    
    def __init__(self, degrees, weights = None, debug = False):
        '''
        Initializes model.
        
        Arguments:
            degrees: List of degrees used in the collection
            weight: None or List of weights
                None: Equal weights are used.
                List: Given weights are used.
        '''
        self.models = []
        self.degrees = np.array(degrees)
        self.debug = debug
        
        if not weights:
            self.weights = np.repeat(1, len(self.degrees))
        else:
            self.weights = np.array(weights)
            
        if self.debug:
            print("Degrees: {}; Weights: {}".format(self.degrees, self.weights))
            
    def fit_poly_reg_model(self, x, y, degree, show = False, ax = None, color="r"):
        '''
        No need to use from otside of this class.
        '''
        poly_features = PolynomialFeatures(degree)
        model = make_pipeline(poly_features, LinearRegression()).fit(x.reshape(-1, 1), y)
        if show:
            print("Coefficients: {}".format(model.steps[1][1].coef_[1:]), "Independent: {}".format(model.steps[1][1].intercept_))
            preds = model.predict(x.reshape(-1, 1))
            sns.set_theme(style="whitegrid")
            if not ax:
                fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(x=x, y=y, ax=ax, color=color)
            plt.plot(x, preds, linestyle="-", color=color)
        return model
    
    def fit(self, x, y, show=False, color="r", ax = None):
        '''
        Fits the model for the data.
        
        Arguments:
            x,y: given data
            show: Plots the model for [min(x), max(x)] interval. Plots also the data.
            color: Color for the graph.
            ax: Axis to plot. If not given, then creates new figure and axis.
        '''
        for degree in self.degrees:
            self.models.append(self.fit_poly_reg_model(x, y, degree = degree))
        if show:
            x_plot = np.linspace(min(x), max(x), 100)
            preds = self.predict(x_plot)
            sns.set_theme(style="whitegrid")
            if not ax:
                fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(x=x, y=y, ax=ax, color=color)
            plt.plot(x_plot, preds, linestyle="-", color=color)
    
    def predict(self, x):
        '''
        Predicts output by given x.
        
        Arguments:
            x: points to predict
        '''
        results = []
        for model in self.models:
            results.append(model.predict(x.reshape(-1, 1)))
        return np.average(np.array(results), axis=0, weights = self.weights)