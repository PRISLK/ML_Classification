# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
#from sklearn.ensemble import RandomForestRegressor


from DataLoader import DataLoader, Columns
import numpy as np

from Submission import Submission


class RandomForest():
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_loader.clean_data()
        self.train_df, self.test_df = self.data_loader.get_dataframes()
        self.submit = Submission('random_forest_53.csv')

    def random_forest(self):
        y = self.train_df[Columns.label]
        X = self.train_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time, Columns.label], axis=1)
        #X = self.train_df.drop([Columns.trip_id, Columns.additional_fare, Columns.meter_waiting_fare, Columns.pickup_time, Columns.drop_time, Columns.pick_lat, Columns.pick_lon, Columns.drop_lat, Columns.drop_lon, Columns.label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

        #rf = RandomForestClassifier(criterion ="gini", oob_score=True, max_depth=2, random_state=0)
        #rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state=0, oob_score=True, verbose=1, max_features=np.sqrt, bootstrap=0.9)
        #rf = RandomForestClassifier(n_estimators = 80)
        rf = RandomForestClassifier(random_state = 0, n_estimators = 80, bootstrap=True, min_samples_split=5, min_samples_leaf=1)
        #rf = RandomForestClassifier()
        
        rf.fit(X_train, y_train)
        
        #param_range = np.arange(100, 250,450, 800)
        
        #train_scoreNum, test_scoreNum = validation_curve(
                                #RandomForestClassifier(),
                                #X = X_train, y = y_train, 
                                #param_name = 'n_estimators', 
                                #param_range = param_range, cv = 3)
        
        print(rf.feature_importances_)
        print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(
            rf.score(X_test, np.ravel(y_test, order='C'))))
        
        predict_df = self.test_df.drop([Columns.trip_id, Columns.pickup_time, Columns.drop_time], axis=1)
        #predict_df = self.test_df.drop([Columns.trip_id,Columns.additional_fare, Columns.meter_waiting_fare, Columns.pickup_time, Columns.drop_time, Columns.pick_lat, Columns.pick_lon, Columns.drop_lat, Columns.drop_lon], axis=1)
        length_of_test = (predict_df.iloc[:, 1].count())
        print("length of test: {}".format(length_of_test))
        
        for i in range(length_of_test):
            input_data = predict_df.iloc[i].values
            input_data_dim = np.expand_dims(input_data, axis=0)
            p = rf.predict(input_data_dim)
            self.submit.write(self.test_df[Columns.trip_id][i], p[0])


if __name__ == "__main__":
    obj = RandomForest()
    obj.random_forest()


