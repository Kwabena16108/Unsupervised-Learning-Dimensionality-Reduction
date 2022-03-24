#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:30:18 2022

@author: dicksonnkwantabisa
"""


import pandas as pd
import numpy as np
from util import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import talib as ta
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, cross_val_predict,
                                      GridSearchCV, RandomizedSearchCV)
import warnings
warnings.filterwarnings('ignore')

IMAGE_DIR = '/Users/dicksonnkwantabisa/Desktop/CS7641-MachineLearning/\
    Unsupervised Learning and Dimensionality Reduction/images/gmm'  # output images directory


#%% DATA PREPROCESSING -- CHURN
df = pd.read_csv("telco_churn_clean.tsv", sep="\t")
clean_df = df.copy()
cat_cols = set()
for c in df.columns:
    if "Yes" in set(df[c].unique()) and df[c].nunique()<=3:
        clean_df[c] = (df[c]=="Yes").astype(int)  
    elif df[c].dtype==object:
        cat_cols.add(c)
    else:
        clean_df[c] = clean_df[c].astype(int)
        
clean_df = pd.get_dummies(clean_df)
clean_df.head()

X_churn, y_churn = clean_df.drop("Churn", axis="columns"), clean_df.Churn
y_churn.nunique() # number of classes


#%% 1. CHURN DATASET

# find a benchmark performance using all data without dimensionality reduction
model = GaussianMixture(random_state=42, n_components = y_churn.nunique(), covariance_type='full',
                        max_iter=1000, n_init=10, init_params='random',)


scaler = MinMaxScaler()
X_churn = scaler.fit_transform(X_churn)

clusters = model.fit_predict(X_churn, )
benchmark(X_churn, y_churn, clusters)

# visualize clusters
viz_clusters_GMM(X=X_churn, y=y_churn, dataset='Churn', method='gmm')

# visualize model complexity 
plot_model_complexity_GMM(x=X_churn, dataset='Churn')

plot_model_complexity_GMM(x=X_etf, dataset='ETF')


#%% 2. CHURN DATASET (ICA REDUCED DATASET)

#---------- plot model complexity

component_range = np.arange(2, X_churn.shape[1]+1)

avg_kurt = [] 
for component in component_range:
    
    ica  = FastICA(n_components=component, max_iter=1000, random_state=42)
    ica.fit(X_churn)
    component_kurt = kurtosis(ica.components_, axis=1, fisher=False)
    avg_kurt.append(np.mean(component_kurt))
    print('k = {} --> average kurtosis = {:.3f}'.format(component, avg_kurt[-1]))
    
# create kurtosis plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))
ax1.plot(component_range, avg_kurt, '-.', markersize=1, label='kurtosis')
set_axis_title_labels(ax1, title='ICA: Choosing number of components (k) by average kurtosis',
                      x_label='Number of Components (k)', y_label='Average Kurtosis')

n_components = 25

model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(X_churn)  # Fit ICA on churn data
# plot axis
x_range = np.arange(1, n_components + 1)
component_kurt = kurtosis(model.components_, axis=1, fisher=False)
ax2.bar(x_range, component_kurt, color='cyan')
ax2.set_xticks(x_range)
set_axis_title_labels(ax2, title='ICA: Kurtosis Distribution of Components',
                      x_label='Independent Component (k)',
                      y_label='Kurtosis')
plt.savefig('images/churn_ica_model_complexity.png')


#------------ fit a GMM clustering algorithm using the ICA components

n_components = 17 # obtained from model complexity plot of ica
ica = FastICA(n_components=n_components, random_state=42,)
X_churn_ica = ica.fit_transform(X_churn)

# find the best cluster components using the reduced data
plot_model_complexity_GMM(x=X_churn_ica, dataset='Churn_ica_reduced')


# find performance using all data with dimensionality reduction
n_components = 6 # obtained from model complexity plot of gmm
model = GaussianMixture(random_state=42, n_components = n_components, covariance_type='full',)
clusters = model.fit_predict(X_churn_ica, )
benchmark(X_churn_ica, y_churn, clusters)

# visualize clusters
viz_clusters_GMM(X=X_churn_ica, y=y_churn, dataset='Churn_ica_reduced', method='gmm')














