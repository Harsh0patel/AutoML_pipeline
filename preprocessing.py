import main as mn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessing():

    def __init__(self):
        self.data = mn.data
        self.df = pd.DataFrame(self.data)
    
    def missing_values(self):
        missing_values = self.df.isnull().sum()
        self.df = self.df.fillna("NAN")
        return missing_values
    
    def data_types(self):
        data_types = self.df.dtypes
        return data_types
    
    def OneHotEncoding(self):
        cat_columns = self.df.select_dtypes(include = ['object']).columns
        OHE = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        df = pd.DataFrame(OHE.fit_transform(self.df[cat_columns]))
        df.columns = OHE.get_feature_names_out(cat_columns)
        self.df = pd.concat([self.df, df], axis = 1)
    
    def Standardization(self):
        scaler = StandardScaler()
        num_columns = self.df.select_dtypes(include= ['float64', 'int64']).columns
        self.df[num_columns] = scaler.fit_transform(self.df[num_columns])

    def copy(self):
        return self.df.copy()
    
    