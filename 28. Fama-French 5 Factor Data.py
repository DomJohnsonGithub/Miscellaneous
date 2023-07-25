import pandas_datareader.data as pdr
from datetime import datetime


def get_fama_french_data(start_date, end_date):
    fama_data = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date, end=end_date)[0]
    return fama_data


start_date = datetime(2000, 1, 3)
end_date = datetime(2021, 6, 24)
factor_data = get_fama_french_data(start_date, end_date)
print(factor_data)
