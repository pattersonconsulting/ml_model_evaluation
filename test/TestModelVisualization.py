import unittest

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.dummy import DummyClassifier


import xgboost as xgb

import ml_valuation
from matplotlib import pyplot as plt

from ml_valuation import model_valuation
from ml_valuation import model_visualization

'''
    Unit testing module for model_visualization.py

    To Run All tests in file:

        (from parent directory)

        python -m unittest test.TestModelVisualization

'''
class TestModelVisualization(unittest.TestCase):


    def test_pr_auc_render(self):

        print("Testing PR Curve Render...")

        model_data_tuples = []

        X, y = make_classification(n_samples=10000, n_features=8, n_redundant=0, n_clusters_per_class=1, weights=[0.85], flip_y=0, random_state=4)
        # split into train/test sets
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

        #print(trainX)

        print("Test Data:\n")
        unique, counts = np.unique(testy, return_counts=True)
        print( dict(zip(unique, counts)) )

        # fit a model
        model = LogisticRegression(solver='lbfgs')
        model.fit(trainX, trainy)
        # predict probabilities
        yhat = model.predict_proba(testX)

        print( "prediction probabilities: " + str(yhat.shape) )

        yhat = yhat[:, 1]


        model_data_tuples.append(tuple((testy, yhat, 'Logistic Regression')))


        model_visualization.plot_pr_curve( model, testX, yhat, testy, "pr_curve_render_test" )







    def test_roc_render(self):

        print("Testing ROC Curve Render...")

        model_data_tuples = []

        X, y = make_classification(n_samples=100, n_features=8, n_redundant=0, n_clusters_per_class=1, weights=[0.85], flip_y=0, random_state=4)
        # split into train/test sets
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

        #print(trainX)

        print("Test Data:\n")
        unique, counts = np.unique(testy, return_counts=True)
        print( dict(zip(unique, counts)) )

        # fit a model
        model = LogisticRegression(solver='lbfgs')
        model.fit(trainX, trainy)
        # predict probabilities
        yhat = model.predict_proba(testX)

        #print( "prediction probabilities: " + str(yhat.shape) )

        yhat = yhat[:, 1]


        model_data_tuples.append(tuple((testy, yhat, 'Logistic Regression')))




        # now let's add the dummy classifier to baseline

        

        clf_dummy = DummyClassifier(strategy='stratified')
        clf_dummy.fit(trainX, trainy)
        #clf_dummy.score(testX, testy)


        dummy_predicted = clf_dummy.predict_proba(testX)
        dummy_predicted = dummy_predicted[:, 1]

        #print( dummy_predicted )
        #print( testy )
        #print("\n\ny_proba TYPE: " + str(type(dummy_predicted)))


        
        model_data_tuples.append(tuple((testy, dummy_predicted, 'Dummy Classifier')))



        """

        # xgBoost

        dtrain = xgb.DMatrix(trainX, label=trainy)
        dtest = xgb.DMatrix(testX, label=testy)


        param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)

        y_pred_xgb = bst.predict(dtest)
        

        #print(confusion_matrix(y, y_pred))   
        model_data_tuples.append(tuple((testy, y_pred_xgb, 'xgBoost')))
        """


        # now lets add gradient boosting

        #from sklearn.preprocessing import Normalizer

        #norm = Normalizer()
        #norm.fit(trainX)
        #X_train_norm = norm.transform(trainX)
        #X_val_norm = norm.transform(testX)   
        
        '''

        GB_params = {'n_estimators':[10, 50, 100, 200]}
        gbc = GradientBoostingClassifier(random_state=42)

        # Instantiate gridsearch using GBC model and search for the best parameters
        gbc_grid = GridSearchCV(gbc, GB_params, cv=3)

        # Fit model to training data
        gbc_grid.fit(trainX, trainy)

        print('Optimized number of estimators: {}'.format(gbc_grid.best_params_.values()))

        # Instantiate GBC with optimal parameters
        gbc_best = GradientBoostingClassifier(**gbc_grid.best_params_, random_state=42)

        # Fit GBC to training data
        gbc_best.fit(trainX, trainy)

        # Evalaute GBC with validation data
        gbc_best_predicted = gbc_best.predict(testX)        


        model_data_tuples.append(tuple((testy, gbc_best_predicted, 'GradientBoostingClassifier')))

        '''


        # now render the ROC Curve

        model_visualization.plot_roc_curves( model_data_tuples, "roc_curve_multiple_models" )



        cost_benefit_matrix = np.array([[175, -150],
                            [0, 0]])


        plt.figure(1)
        fig, ax = plt.subplots(1,1,figsize = (10,10))

        model_visualization.plot_profit_curves( cost_benefit_matrix, model_data_tuples, fig, ax, "profit_curve_multiple_models", total_budget=1500 )




if __name__ == '__main__':
    unittest.main()