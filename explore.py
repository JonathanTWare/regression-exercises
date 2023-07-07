import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






def plot_variable_pairs(df, sample_size=100000, variables=None):
    if variables is None:
        variables = df.columns

    sns.set(style="ticks")
    
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=123)
    else:
        df_sample = df

    
    sns.pairplot(df_sample[variables], kind="reg", plot_kws={'line_kws': {'color': 'red'}})
    plt.show()
    

def plot_categorical_and_continuous_vars(df):

    categorical_col = 'county_code'
    continuous_cols = ['bedroom_count','bathroom_count','calc_sqr_feet', 'tax_value' , 'yearbuilt', 'taxamount']
    for continuous_col in continuous_cols:
        plt.figure(figsize=(10, 4))
        
        
        plt.subplot(131)
        sns.boxplot(data = df.sample(frac = 0.10), x=categorical_col, y=continuous_col)
        plt.title('Box Plot')
        
        plt.subplot(132)
        sns.violinplot(data = df.sample(frac = 0.10),x=categorical_col, y=continuous_col)
        plt.title('Violin Plot')
        
        plt.subplot(133)
        sns.scatterplot(data = df.sample(frac = 0.10), x=categorical_col, y=continuous_col)
        plt.title('Scatter Plot')

    