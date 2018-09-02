# Machine Learning Python K-Means Clustering

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
1.	Randomly pick K cluster centers(centroids). 
2.	Assign each point to closest center. By geometry join 2 centroid and devide by perpendicular line to identify closest.
3.	Select the new centroid by taking the average of Euclidean distances of all the points assigned to that cluster.
4.	Repeat 2 and 3 until clustor assignment stop changing.

## Screenshot</br>
<img src="images/K-Means Clustering 2018-08-25 03-01-14.png">

## Data Story:
This data is 2010 census data. After clustering using K-Means we plotted data which clearly shows most of the population lies around 38 of median age. As population grows median ages is also changing and its coming down. This model gives idea, if business need to manufacture products according to ~age of 40 for 20K population, ~age of 39 for next 18K population, ~age of 38 for 18K, ~age of 36 for 24K and age of 32 for 34K.

## Code: ViewController.swift
```
# K-Means Clustering

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

# Using the elbow method to get the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method showing the optimal K')
plt.xlabel('K - Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the sample dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Scatter chart of the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'gold', label = 'Cluster A')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'skyblue', label = 'Cluster B')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'orchid', label = 'Cluster C')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'mediumspringgreen', label = 'Cluster D')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'c', label = 'Cluster E')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'red', label = 'Centroids')
plt.title('Clusters of Population')
plt.xlabel('Total Population')
plt.ylabel('Median Age')
plt.legend()
plt.show()
```

## Thank You

