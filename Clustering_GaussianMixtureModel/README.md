# Machine Learning Python Python Gaussian Mixture Model Clustering Clustering

This example explains k-means clustering with Python 3, pandas and scikit-learn on Jupyter Notebook.
## Requirements
To use this example you need Python 3 and latest versions of pandas and scikit-learn. I used Anaconda distribution to install.

## Data Set:
https://catalog.data.gov/dataset/2010-census-populations-by-zip-code

## ML life-cycle:
1.	Business objective connected to it.
2.	Data set, wrangle and prepare it.
3.	What the data is saying

## Algorithm
1.	A Gaussian mixture model (GMM) attempts to find a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset.

## Screenshot</br>
<img src="images/GaussianMixtureModel Clustering 2018-09-02 14-55-22.png">

## Data Story:
This data is 2010 census data. After clustering using GMM we plotted data which clearly shows most of the population lies around 38 of median age. As population grows median ages is also changing and its coming down. This model gives idea, if business need to manufacture products according to ~age of 40 for 20K population, ~age of 39 for next 18K population, ~age of 38 for 18K, ~age of 36 for 24K and age of 32 for 34K.

## Code: ViewController.swift
```
# GMM Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('2010_Census_Populations.csv')
#Data Prepare
# Replacing 0 to NaN
dataset[['Total Population','Median Age']] = dataset[['Total Population','Median Age']].replace(0, np.NaN)
X = dataset.iloc[:, [1, 2]].values
#print(X)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

print(X)

# Fitting GMM to the dataset
from sklearn.mixture import GaussianMixture as GMM
#gmm = GMM(n_components=4, covariance_type='full', random_state=42).fit(X)
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)

# Scatter chart of the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
```

## Thank You

