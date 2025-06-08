import pandas as pd
from io import StringIO
# Sample file path for testing purposes
# file_path = "retail_data_sample.csv"

class load_file():
    def __init__(self, file_path):
        self.file_path = file_path

    #file path form front end with max size 5MB
    def Load(self):
        try:
            with open(self.file_path, 'r') as file:
                data = file.read()
                if len(data) > 5 * 1024 * 1024:
                    return "File size exceeds 5MB limit."
                else:
                    buffer = StringIO(data)
                    df = pd.read_csv(buffer)
                    return df
            file.close()
        except FileNotFoundError:
            return "File not found."