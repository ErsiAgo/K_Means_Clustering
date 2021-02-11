#
from dtaidistance import dtw_visualisation as dtwvis
import math
from sklearn.cluster import KMeans
import xlsxwriter
import os
from itertools import combinations
from dtaidistance import dtw
from numpy.core._multiarray_umath import ndarray
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import sympy
# Retrieve current working directory

cwd = os.getcwd()
cwd

# List directory
os.listdir('.')

# Import the data
df = pd.read_excel(r'C:\Users\EA\Desktop\Paper\Results.xlsx')
#print(df)

# timeseries = np.array([
#      pd.Series(df['0'] + 0.307808),
#      pd.Series(df['1'] - 68.187345),
#      pd.Series(df['2'] - 59.33976)])
print(df)

timeseries = np.array([
      pd.Series(df['A12']),
      pd.Series(df['A13']),
      pd.Series(df['A23'])])

# create kmeans object
kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
kmeans.fit(timeseries)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(timeseries)
plt.scatter(timeseries[y_km ==0,0], timeseries[y_km == 0,1], s=100, c='red')
plt.scatter(timeseries[y_km ==1,0], timeseries[y_km == 1,1], s=100, c='black')
plt.show()
############################################


