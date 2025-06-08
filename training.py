import main as mn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Training():
    def __init__(self):
        self.data = mn.data
        self.df = pd.DataFrame(self.data)
        # Seperating features and target variables
        X = self.df.drop(columns = ['target'])
        Y = self.df['target']
        self.X = X
        self.Y = Y

    def coorelation(self):
        # Calculating correlation
        corr = self.X.corr(self.Y)
        corr = corr.abs().sort_values(ascending = False)
        return corr

    def split_data(self):
        # Train test split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size = 0.8, random_state = 42)
        return x_train, x_test, y_train, y_test
    
class Models():
    def __init__(self):
        train = Training()
        self.x_train, self.x_test, self.y_train, self.y_test = train.split_data()
    
    def linear_regression(self):
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy

    def logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return model, accuracy, matrix

    def regide_regression(self):
        model = Ridge()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def lasso_regression(self):
        model = Lasso()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def elastic_net_regression(self):
        model = ElasticNet()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def support_vector_regression(self):
        model = SVR()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def KNN_regression(self):
        model = KNeighborsRegressor()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def decision_tree_regression(self):
        model = DecisionTreeRegressor()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return model, accuracy
    
    def SVM_classification(self):
        model = SVC()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return model, accuracy, matrix
    
    def KNN_classifier(self):
        model = KNeighborsClassifier()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return model, accuracy, matrix
    
    def decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return model, accuracy, matrix

    def naive_bayes_classifier(self):
        model = GaussianNB()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = classification_report(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        return model, accuracy, matrix
    
    def cross_validation(self, model):
        score = cross_val_score(model, self.x_train, self.y_train, cv = 5)
        return score.mean()
    
    def dump_model(self, model, filename):
        import pickle
        with open(filename, 'wb') as file:
            pickle.dump(model, file)