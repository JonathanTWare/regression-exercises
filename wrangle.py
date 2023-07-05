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
from sklearn.impute import SimpleImputer


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


def prep_zillow_data(df):
    df.replace(r'^\s*$', np.nan, regex=True)
    df.dropna()
    df.drop_duplicates(keep= False)

    

    bin_cat = [0, 500000, 1000000, 1500000, 2000000, 5000000, 10000000, 25000000, 50000000, 75000000, 100000000]
    df['value_cat'] = pd.cut(df['taxvaluedollarcnt'], bins=bin_cat, labels=False)
      
    
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

