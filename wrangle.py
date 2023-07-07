#import libraries
import pandas as pd
import numpy as np
from env import get_db_url
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from pydataset import data
from env import get_db_url
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def new_zillow_data():
   
    conn = get_db_url('zillow')

    query = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            WHERE propertylandusetypeid = 261;  
            '''

    
    df = pd.read_sql(query, conn)
    return df
    
def get_zillow_data():
    if os.path.isfile('zillow_df.csv'):
        df = pd.read_csv('zillow_df.csv', index_col = 0)
        

    else:

        df = new_zillow_data()
        df.to_csv('zillow_df.csv')
        
    return df
    

# -----------------------------prep--------------------------------
def scale_data(train, validate, test):
    
    numeric_cols = ['bedroom_count','bathroom_count','calc_sqr_feet','yearbuilt','taxamount']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    
    train_scaled[numeric_cols] = scaler.transform(train[numeric_cols])
    validate_scaled[numeric_cols] = scaler.transform(validate[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test[numeric_cols])
    
    return train_scaled, validate_scaled, test_scaled


def prep_zillow_data(df):
    
    df = df.dropna()
    new_columns = {
        'bedroomcnt': 'bedroom_count',
        'bathroomcnt': 'bathroom_count',
        'calculatedfinishedsquarefeet': 'calc_sqr_feet',
        'taxvaluedollarcnt': 'tax_value',
        'fips': 'county_code'
    }
    df = df.rename(columns=new_columns)
    df_encoded = pd.get_dummies(df['county_code'], prefix='is_county_code')
    df = pd.concat([df, df_encoded], axis=1)

    


    df = df[
        (df['bedroom_count'] <= 5) &
        (df['bedroom_count'] >=3) &
        (df['bathroom_count'] <= 4) &
        (df['bathroom_count'] >= 2) &
        (df['calc_sqr_feet'] <= 9000) &
        (df['yearbuilt'] >=1900) &
        (df['taxamount'] <=120000)
    ]
  
    
    train, validate, test = split_zillow_data(df)
    
    return train, validate, test

   

def split_zillow_data(df):
  
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123) 
                                       

    
    return train, validate, test

def wrangle_zillow():
    df = get_zillow_data()
    train, validate, test = prep_zillow_data(df)
    return train, validate, test

