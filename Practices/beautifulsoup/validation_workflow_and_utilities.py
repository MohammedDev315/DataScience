import pandas as pd
import numpy as np

df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', header=None)

print(df.columns)
columns= ['symboling','normalized-losses','make','fuel-type',
          'aspiration','num-of-doors','body-style','drive-wheels',
          'engine-location','wheel-base','length','width','height',
          'curb-weight','engine-type','num-of-cylinders','engine-size',
          'fuel-system','bore','stroke','compression-ratio','horsepower',
          'peak-rpm','city-mpg','highway-mpg','price']
df.columns=columns

print(df.columns)