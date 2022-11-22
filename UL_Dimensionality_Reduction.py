#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:55:07 2022

@author: dixondomfeh
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
from sklearn.model_selection import train_test_split
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, cross_val_predict,
                                      GridSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

IMAGE_DIR = '/Users/dicksonnkwantabisa/Desktop/CS7641-MachineLearning/Unsupervised Learning and Dimensionality Reduction/images/'  # output images directory

#%% Data Preprocessing (CHURN DATASET)
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

#---------------- (NAIVE model) find a benchmark performance using all data without dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_churn.nunique(),)
clusters = model.fit_predict(X_churn, )
print('\nGMM benchmark performance using all the data:')
benchmark(X_churn, y_churn, clusters)

model = GaussianMixture(random_state=42, n_components = y_churn.nunique(),)
clusters = model.fit_predict(X_churn, )
print('\nGMM benchmark performance using all the data:')
benchmark(X_churn, y_churn, clusters)


# visualize clusters
viz_clusters_KMeans(X=X_churn, y=y_churn, dataset='Churn_all_data_ground_truth', method='kmeans')
viz_clusters_GMM(X=X_churn, y=y_churn, dataset='Churn_all_data_ground_truth', method='gmm')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_churn, dataset='Churn_all_data_ground_truth')
plot_model_complexity_GMM(x=X_churn, dataset='Churn_all_data_ground_truth')


kmeans_orig_clusters = kmeans_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_clusters=2, dataset='churn')

gmm_orig_clusters = gmm_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_components=2, dataset='churn')

#%% 2. CHURN DATASET (ICA REDUCED DATASET)

#---------- experiment (using un-transformed data)
            # use this code to choose the optimal hyperparameters for
            # Kmeans : number of clusters
            # GMM : number of components
"""A Gaussian mixture model is a probabilistic model that assumes all 
the data points are generated from a mixture of a finite number of 
Gaussian distributions with unknown parameters. One can think of 
mixture models as generalizing k-means clustering to incorporate 
information about the covariance structure of the data as well 
as the centers of the latent Gaussians.

useful link : https://scikit-learn.org/stable/modules/mixture.html

    """
# scaled data using MinmaxScaler()
x_train, x_test, y_train, y_test=split_dataset(X=X_churn, y=y_churn, dataset='churn')
# visualize clusters of scaled training data
viz_clusters_KMeans(X=x_train, y=y_train, dataset='churn_scaled_data', method='kmeans')
viz_clusters_GMM(X=x_train, y=y_train, dataset='churn_scaled_data', method='gmm')

kmeans_orig_clusters = kmeans_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_clusters=2, dataset='churn')

gmm_orig_clusters = gmm_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_components=2, dataset='churn')


#------------ check model complexity to determine the optimal n_clusters for kmeans and n_components for GMM
plot_model_complexity_KMeans(x=x_train, dataset='churn_scaled_data') 
plot_model_complexity_GMM(x=x_train, dataset='churn_scaled_data')   


#---------- plot model complexity (choosing the number of components for ICA)

component_range = np.arange(2, x_train.shape[1]+1)
avg_kurt = [] 
for component in component_range:
    ica  = FastICA(n_components=component, max_iter=1000, random_state=42)
    ica.fit(x_train)
    component_kurt = kurtosis(ica.components_, axis=1, fisher=False)
    avg_kurt.append(np.mean(component_kurt))
    print('k = {} --> average kurtosis = {:.3f}'.format(component, avg_kurt[-1]))    
# create kurtosis plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))
ax1.plot(component_range, avg_kurt, '-.', markersize=1, label='kurtosis')
set_axis_title_labels(ax1, title='ICA: Choosing number of components (k) by average kurtosis',
                      x_label='Number of Components (k)', y_label='Average Kurtosis')

n_components = x_train.shape[1] # number of features
model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(x_train)  # Fit ICA on churn data
x_range = np.arange(1, n_components + 1) # plot axis
component_kurt = kurtosis(model.components_, axis=1, fisher=False)
ax2.bar(x_range, component_kurt, color='cyan')
ax2.set_xticks(x_range)
set_axis_title_labels(ax2, title='ICA: Kurtosis Distribution of Components',
                      x_label='Independent Component (k)',
                      y_label='Kurtosis')
plt.savefig('images/churn_ica_model_complexity.png')


#------------ fit a KMeans clustering algorithm using the ICA components

n_components = np.argmax(np.array(avg_kurt))+2 # obtained from model complexity plot for ICA
ica = FastICA(n_components=n_components, random_state=42,)
x_churn_ica = ica.fit_transform(x_train)
ica_experiment = experiment_ica(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='churn',
                                n_components=n_components,# replaced with optimal components
                                name='ica_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=ica_experiment[0], dataset='churn_ica_reduced')

n_clusters = 5 # this is obtained from model complexity plot of kmeans
kmeans_ica_clusters = kmeans_experiment(x_train=ica_experiment[0], 
                                        x_test=ica_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='churn_ica_reduced')




#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=ica_experiment[0], dataset='churn_ica_reduced')

n_components = 15 # obtained from model complexity plot for GMM
gmm_ica_clusters = gmm_experiment(x_train=ica_experiment[0], 
                                   x_test=ica_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='churn_ica_reduced')

#%% 3. CHURN DATASET (PCA REDUCED DATASET)

 
#---------- plot model complexity
pca = PCA(svd_solver='auto',n_components = 0.99, random_state=42) # 95% variance explanation
pca.fit(x_train)

# compute total explained variance by number of components
n_components = 2
explained_variance = np.sum(pca.explained_variance_ratio_[:n_components])
print('Explained variance [ n compents = {}] = {:.3f}'.format(n_components, explained_variance))

# create plots
component_range = np.arange(1, len(pca.explained_variance_ratio_)+1)
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
# plot the cumulative dsitribution of explained variance
ax1.plot(component_range, np.cumsum(pca.explained_variance_ratio_),marker='o',color='b', linestyle='--')
ax1.axhline(y=0.95, color='r', linestyle='-')
ax1.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=12)
set_axis_title_labels(ax1, title='PCA: Choosing number of components (k) by explained variance ratio',
                      x_label='Number of components (k)', y_label='Cumulative variance (%)')
# plot explained variance ratio or eigenvalue distirbution
ax2.bar(component_range,pca.explained_variance_ratio_, color='cyan')
set_axis_title_labels(ax2, title='PCA - Eigenvalues Distribution', 
                      x_label='Number of components (k)', y_label='Variance (%)')
save_figure_tight('{}_pca_model_complexity'.format('churn'))


#------------ fit a KMeans clustering algorithm using the PCA components

n_components = 16 # obtained from model complexity plot
pca = PCA(n_components=n_components, random_state=42,)
pca.fit(x_train)
x_churn_pca = pca.fit_transform(x_train)
pca_experiment = experiment_pca(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='churn',
                                n_components=n_components,# replaced with optimal components
                                name='pca_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=pca_experiment[0], dataset='churn_pca_reduced')

n_clusters = 4 # this is obtained from model complexity plot of kmeans
kmeans_pca_clusters = kmeans_experiment(x_train=pca_experiment[0], 
                                        x_test=pca_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='churn_pca_reduced')


#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=pca_experiment[0], dataset='churn_pca_reduced')

n_components = 19 # obtained from model complexity plot for GMM
gmm_pca_clusters = gmm_experiment(x_train=pca_experiment[0], 
                                   x_test=pca_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='churn_pca_reduced')



#%% 4. CHURN DATASET (Kernel PCA REDUCED DATASET)

#---------- plot model complexity

n_components = 2
component_range = np.arange(1, X_churn.shape[1]+1)
kernels = ['rbf', 'sigmoid', 'poly', 'cosine']

# create plots
fig, ax = plt.subplots(4, 2, figsize=(15,30))
ax = ax.ravel()
for i, kernel in enumerate(kernels):
    kpca = KernelPCA(n_components=x_train.shape[1], kernel=kernel, random_state=42, n_jobs=-1)
    kpca.fit(x_train)
    explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
    explained_variance = np.sum(explained_variance_ratio[:n_components])
    print('When kernel = {} - Explained variance [n components = {}]= {:.3f}'.format(kernel,
                                                                                n_components,
                                                                                explained_variance))

    # Plot histogram of the cumulative explained variance ratio
    ax[2*i].plot(component_range, np.cumsum(explained_variance_ratio), marker='o',color='b', linestyle='--', label=kernel)
    ax[2*i].axhline(y=0.95, color='r', linestyle='-')
    ax[2*i].text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    # Set title, labels and legend
    ax[2*i].legend(loc='best')
    set_axis_title_labels(ax[2*i], title='KPCA - Choosing components (k) with the Variance method',
                                x_label='Number of components (k)', y_label='Cumulative Variance (%)')

    ax[2*i+1].bar(component_range, explained_variance_ratio, color='cyan', label=kernel)

    ax[2*i+1].legend(loc='best')
    set_axis_title_labels(ax[2*i+1], title='KPCA - Eigenvalues distributions',
                                x_label='Number of components k', y_label='Variance (%)')
save_figure_tight('{}_kpca_model_complexity'.format('churn'))

# optimal choices:
        # rbf = 18 components
        # poly = 17 components
        # sigmoid = 16 components
        # cosine = 16 components
        
#------------ fit a KMeans clustering algorithm using the Kernel PCA components

n_components = 16 # obtained from model complexity plot (sigmoid kernel)
kpca = KernelPCA(n_components=n_components, kernel='sigmoid', random_state=42, n_jobs=-1)
kpca.fit(x_train) # fit model
x_churn_kpca = kpca.transform(x_train) 
kpca_experiment = experiment_kpca(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='churn',
                                n_components=n_components,# replaced with optimal components
                                name='churn_kpca_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=kpca_experiment[0], dataset='churn_kpca_reduced')

n_clusters = 4 # this is obtained from model complexity plot of kmeans
kmeans_kpca_clusters = kmeans_experiment(x_train=kpca_experiment[0], 
                                        x_test=kpca_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='churn_kpca_reduced')

#------------ fit a GMM clustering algorithm using the kernel PCA components

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=kpca_experiment[0], dataset='churn_kpca_reduced')

n_components = 18 # obtained from model complexity plot for GMM
gmm_kpca_clusters = gmm_experiment(x_train=kpca_experiment[0], 
                                   x_test=kpca_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='churn_kpca_reduced')


#%% 5. CHURN DATASET (Randomized Projection REDUCED DATASET)


def create_viz(x_train, y, abs_diff, dataset, name):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    
    plt.subplot(131)
    df = pd.DataFrame(x_train[:, :2], columns=['gauss_prj1', 'gauss_prj2'])
    df['y'] = y_train
    colors = sns.color_palette('hls', len(np.unique(df['y'].dropna())))
    sns.scatterplot(x='gauss_prj1', y='gauss_prj2', hue='y', palette=colors, data=df, legend='full', alpha=0.3)
    # Annotate standard deviation arrows for the two components
    plt.annotate('', xy=(np.std(df['gauss_prj1']), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df['gauss_prj2'])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    # Set title and axes limits
    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df['gauss_prj1']))
    ylim = 1.1 * np.max(np.abs(df['gauss_prj2']))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    plt.subplot(132)
    plt.imshow(abs_diff)
    plt.colorbar()
    plt.title('Visualization of absolute differences')
    
    plt.subplot(133)
    ax = plt.hist(abs_diff.flatten())
    plt.title('Histogram of absolute differences')
    
    fig.subplots_adjust(wspace=.3)


#---------- plot model complexity

reduction_dim_gauss = []
eps_arr_gauss = []
mean_abs_diff_gauss = []
n_component_range = np.arange(5, 21,5)

for eps in np.arange(0.1, 0.999, 0.2):
    for n in n_component_range:
        #min_dim = johnson_lindenstrauss_min_dim(n_samples=x_train.shape[0], eps=eps)
        #if min_dim > x_train.shape[0]:
         #   continue
        gauss_proj = GaussianRandomProjection(n_components=n,random_state=42,eps=eps,)
        X_transform = gauss_proj.fit_transform(x_train)
        dist_raw = euclidean_distances(x_train)
        dist_transform = euclidean_distances(X_transform)
        abs_diff_gauss = abs(dist_raw - dist_transform) 
    
        create_viz(x_train=X_transform, y=y_train, abs_diff=abs_diff_gauss, dataset='churn', name='gauss_proj')
        plt.suptitle('eps = ' + '{:.2f}'.format(eps) + ', n_components = ' + str(X_transform.shape[1]))

        reduction_dim_gauss.append(100-X_transform.shape[1]/X_transform.shape[1]*100)
        eps_arr_gauss.append(eps)
        mean_abs_diff_gauss.append(np.mean(abs_diff_gauss.flatten()))
    
 
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
plt.subplot(121)
plt.plot(eps_arr_gauss, mean_abs_diff_gauss, marker='o', c='g')
plt.xlabel('eps')
plt.ylabel('Mean absolute difference')

plt.subplot(122)
plt.plot(eps_arr_gauss, reduction_dim_gauss, marker = 'o', c='m')
plt.xlabel('eps')
plt.ylabel('Percentage reduction in dimensionality')

fig.subplots_adjust(wspace=.4) 
plt.suptitle('Assessing the Quality of Gaussian Random Projections')
plt.show()

# Randomized projections don't work well on small features
# Our dataset is too small to determine an appropriate minimum dimension.


# ##################### #########   run rp several times and see what happens  ##############################
rand_gauss(x=x_train, dataset='churn')     
########################################################################

#------------ fit a KMeans clustering algorithm using the RGP components

n_components = 14 # obtained from model complexity plot for RGP
eps = 0.9
rgp = GaussianRandomProjection(n_components=n_components,random_state=42,eps=eps,)
x_churn_rgp = rgp.fit_transform(x_train)
rgp_experiment = experiment_rgp(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='churn',
                                n_components=n_components,
                                eps=eps,
                                name='rgp_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=rgp_experiment[0], dataset='churn_rgp_reduced')

n_clusters = 2 # this is obtained from model complexity plot of kmeans
kmeans_rgp_clusters = kmeans_experiment(x_train=rgp_experiment[0], 
                                        x_test=rgp_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='churn_rgp_reduced')


#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=rgp_experiment[0], dataset='churn_rgp_reduced')

n_components = 20 # obtained from model complexity plot for GMM
gmm_rgp_clusters = gmm_experiment(x_train=rgp_experiment[0], 
                                   x_test=rgp_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='churn_rgp_reduced')



#%% Data Preprocessing (ETF DATASET)

# download 30 years of daily data from Yahoo Finance

tickers = ['SPY']
df = yf.download(" ".join(tickers), period='30y', interval='1d', group_by='tickers')
df.reset_index(inplace=True)
# extract time features
df['year'] = df.loc[:, 'Date'].dt.year
df['month'] = df.loc[:, 'Date'].dt.month
df['day_of_week'] = df.loc[:, 'Date'].dt.day_of_week
df['week'] = df.loc[:, 'Date'].dt.isocalendar().week

#----------- Feature engineering

# High - Low
df['H-L'] = df.High - df.Low
# Open - Close
df['O-C'] = df.Open - df.Close
# Correlation
df['volume_by_adv20'] = df.Volume/df.Volume.rolling(20).mean()

# Pass previous High, Low, Close for the alogoritm to have a sense of volatility in the past
df['Prev_High'] = df['High'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_Close'] = df['Close'].shift(1)

# Create columns 'OO' with the difference between the current minute's open and last day's open
df['OO'] = df['Open']-df['Open'].shift(1)

# Create columns 'OC' with the difference between the current minute's open and prevo
df['OC'] = df['Open']-df['Prev_Close']

################# TECHNICAL INDICATORS ######################
def get_momentum(prices, window):
    momentum = prices / prices.shift(window) - 1
    return momentum

def get_bb(prices, window):
    rm = prices.rolling(window).mean()
    rstd = prices.rolling(window).std()
    bbp = (prices - rm) / 2 * rstd
    return bbp 

def get_psma(prices, window):
    rm = prices.rolling(window).mean()
    psma = prices.divide(rm, axis=0) - 1
    return psma 

def get_pema(prices, window):
    ema = prices.ewm(window).mean()
    pema = prices.divide(ema, axis=0) - 1
    return pema


# Create a lookback period(n) = 10-days
n=10

# 1. Relative Strength Index (RSI)
df['RSI'] = ta.RSI(df['Adj Close'].shift(-1), timeperiod=n)

# 2. SMA
df['SMA'] = df['Adj Close'].shift(1).rolling(window=n).mean()

# 3. Correlation between Adj Close and SMA
df['Corr'] = df['Adj Close'].shift(1).rolling(window=n).corr(df.SMA.shift(1))

# 4. Parabolic SAR (stop and reverse)
df['SAR'] = ta.SAR(df.High.shift(1), df.Low.shift(1), 0.2, 0.2)

# 5. ADX (Average directional movement index)
df['ADX'] = ta.ADX(df.High.shift(1), df.Low.shift(1), df.Open, timeperiod=n)

# 6. NATR (Normalized average true range)
df['NATR'] = ta.NATR(df.Low,df.High,df.Close, timeperiod=n)

# 7. Bollinger bands
df['BB'] = get_bb(df['Adj Close'], window=n)

# 8. Price / SMA
df['PSMA'] = get_psma(df['Adj Close'], window=n)

# 9. Price / EMA
df['PEMA'] = get_pema(df['Adj Close'], window=n)

# 10. Momentum
df['MOM'] = get_momentum(df['Adj Close'], window=n)
df['MOM10'] = get_momentum(df['Adj Close'], window=20)
df['MOM30'] = get_momentum(df['Adj Close'], window=30)
df['MOM40'] = get_momentum(df['Adj Close'], window=40)


# returns
df['ret1'] = df['Adj Close'].pct_change()

df['ret5'] = df['ret1'].rolling(5).sum()
df['ret10'] = df['ret1'].rolling(10).sum()
df['ret20'] = df['ret1'].rolling(20).sum()
df['ret40'] = df['ret1'].rolling(40).sum()

# One-day future returns
df['retFut1'] = df.ret1.shift(-1)

# calculate past returns to help algorithm understand the trends in the last n periods
for i in range(1, n):
    df['return%i' % i] = df.retFut1.shift(i)

# Change the value of 'Corr' to -1 if it is less than -1
df.loc[df.Corr < -1, 'Corr'] = -1

# Change the value of 'Corr' to 1 if it is greater than 1
df.loc[df['Corr'] > 1, 'Corr'] = 1

# Drop the NaN values
df = df.dropna()


# Train and test data
t = .8
split = int(t*len(df))
df['Signal'] = 0
df.loc[df.retFut1 > df.retFut1[:split].quantile(q=0.66), 'Signal'] = 1 # BUY signal
df.loc[df.retFut1 < df.retFut1[:split].quantile(q=0.34), 'Signal'] = 2 # SELL signal

X_etf = df.drop(['Date','Open','Close','Adj Close','Signal', 'High','Low', 'Volume', 'retFut1'], axis=1)
y_etf = df['Signal']


# Original matrix to be used to calculate reconstruction error
# X_ETF = df.drop(['Date','Open','Close','Adj Close','Signal', 'High','Low', 'Volume', 'retFut1'], axis=1)



#%% 1. ETF DATASET

# scaled data using MinmaxScaler()
x_train, x_test, y_train, y_test=time_series_data_split(X=X_etf, y=y_etf, dataset='etf')
# visualize clusters of scaled training data
viz_clusters_KMeans(X=x_train, y=y_train, dataset='etf_scaled_data', method='kmeans')
viz_clusters_GMM(X=x_train, y=y_train, dataset='etf_scaled_data', method='gmm')

kmeans_orig_clusters = kmeans_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_clusters=3, dataset='etf')

gmm_orig_clusters = gmm_experiment(x_train=x_train, x_test=x_test, y_train=y_train,
                                         y_test=y_test, n_components=3, dataset='etf')


#%% 2. ETF DATASET (ICA REDUCED DATASET)


#---------- plot model complexity (choosing the number of components for ICA)

component_range = np.arange(2, x_train.shape[1]+1)
avg_kurt = [] 
for component in component_range:
    ica  = FastICA(n_components=component, max_iter=1000, random_state=42)
    ica.fit(x_train)
    component_kurt = kurtosis(ica.components_, axis=1, fisher=False)
    avg_kurt.append(np.mean(component_kurt))
    print('k = {} --> average kurtosis = {:.3f}'.format(component, avg_kurt[-1]))    
# create kurtosis plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))
ax1.plot(component_range, avg_kurt, '-.', markersize=1, label='kurtosis')
set_axis_title_labels(ax1, title='ICA: Choosing number of components (k) by average kurtosis',
                      x_label='Number of Components (k)', y_label='Average Kurtosis')

n_components = x_train.shape[1] # number of features
model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(x_train)  # Fit ICA on churn data
x_range = np.arange(1, n_components + 1) # plot axis
component_kurt = kurtosis(model.components_, axis=1, fisher=False)
ax2.bar(x_range, component_kurt, color='cyan')
ax2.set_xticks(x_range)
set_axis_title_labels(ax2, title='ICA: Kurtosis Distribution of Components',
                      x_label='Independent Component (k)',
                      y_label='Kurtosis')
plt.savefig('images/etf_ica_model_complexity.png')


#------------ fit a KMeans clustering algorithm using the ICA components

n_components = np.argmax(np.array(avg_kurt))+2 # obtained from model complexity plot for ICA
ica = FastICA(n_components=n_components, random_state=42,)
x_etf_ica = ica.fit_transform(x_train)
ica_experiment = experiment_ica(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='etf',
                                n_components=n_components,# replaced with optimal components
                                name='ica_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=ica_experiment[0], dataset='etf_ica_reduced')

n_clusters = 3 # this is obtained from model complexity plot of kmeans
kmeans_ica_clusters = kmeans_experiment(x_train=ica_experiment[0], 
                                        x_test=ica_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='etf_ica_reduced')


#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=ica_experiment[0], dataset='etf_ica_reduced')

n_components = 19 # obtained from model complexity plot for GMM
gmm_ica_clusters = gmm_experiment(x_train=ica_experiment[0], 
                                   x_test=ica_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='etf_ica_reduced')



#%% 3. ETF DATASET (PCA REDUCED DATASET)

 
#---------- plot model complexity
pca = PCA(svd_solver='auto',n_components = 0.99, random_state=42) # 95% variance explanation
pca.fit(x_train)

# compute total explained variance by number of components
n_components = 2
explained_variance = np.sum(pca.explained_variance_ratio_[:n_components])
print('Explained variance [ n compents = {}] = {:.3f}'.format(n_components, explained_variance))

# create plots
component_range = np.arange(1, len(pca.explained_variance_ratio_)+1)
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
# plot the cumulative dsitribution of explained variance
ax1.plot(component_range, np.cumsum(pca.explained_variance_ratio_),marker='o',color='b', linestyle='--')
ax1.axhline(y=0.95, color='r', linestyle='-')
ax1.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=12)
set_axis_title_labels(ax1, title='PCA: Choosing number of components (k) by explained variance ratio',
                      x_label='Number of components (k)', y_label='Cumulative variance (%)')
# plot explained variance ratio or eigenvalue distirbution
ax2.bar(component_range,pca.explained_variance_ratio_, color='cyan')
set_axis_title_labels(ax2, title='PCA - Eigenvalues Distribution', 
                      x_label='Number of components (k)', y_label='Variance (%)')
save_figure_tight('{}_pca_model_complexity'.format('etf'))


#------------ fit a KMeans clustering algorithm using the PCA components

n_components = 9 # obtained from model complexity plot
pca = PCA(n_components=n_components, random_state=42,)
pca.fit(x_train)
x_etf_pca = pca.fit_transform(x_train)
pca_experiment = experiment_pca(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='churn',
                                n_components=n_components,# replaced with optimal components
                                name='pca_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=pca_experiment[0], dataset='etf_pca_reduced')

n_clusters = 4 # this is obtained from model complexity plot of kmeans
kmeans_pca_clusters = kmeans_experiment(x_train=pca_experiment[0], 
                                        x_test=pca_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='etf_pca_reduced')


#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=pca_experiment[0], dataset='etf_pca_reduced')

n_components = 18 # obtained from model complexity plot for GMM
gmm_pca_clusters = gmm_experiment(x_train=pca_experiment[0], 
                                   x_test=pca_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='etf_pca_reduced')



#%% 4. ETF DATASET (Kernel PCA REDUCED DATASET)

#---------- plot model complexity

n_components = 2
component_range = np.arange(1, x_train.shape[1]+1)
kernels = ['rbf', 'sigmoid', 'poly', 'cosine']

# create plots
fig, ax = plt.subplots(4, 2, figsize=(15,30))
ax = ax.ravel()
for i, kernel in enumerate(kernels):
    kpca = KernelPCA(n_components=x_train.shape[1], kernel=kernel, random_state=42, n_jobs=-1)
    kpca.fit(x_train)
    explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
    explained_variance = np.sum(explained_variance_ratio[:n_components])
    print('When kernel = {} - Explained variance [n components = {}]= {:.3f}'.format(kernel,
                                                                                n_components,
                                                                                explained_variance))

    # Plot histogram of the cumulative explained variance ratio
    ax[2*i].plot(component_range, np.cumsum(explained_variance_ratio), marker='o',color='b', linestyle='--', label=kernel)
    ax[2*i].axhline(y=0.95, color='r', linestyle='-')
    ax[2*i].text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    # Set title, labels and legend
    ax[2*i].legend(loc='best')
    set_axis_title_labels(ax[2*i], title='KPCA - Choosing components (k) with the Variance method',
                                x_label='Number of components (k)', y_label='Cumulative Variance (%)')

    ax[2*i+1].bar(component_range, explained_variance_ratio, color='cyan', label=kernel)

    ax[2*i+1].legend(loc='best')
    set_axis_title_labels(ax[2*i+1], title='KPCA - Eigenvalues distributions',
                                x_label='Number of components k', y_label='Variance (%)')
save_figure_tight('{}_kpca_model_complexity'.format('etf'))

# optimal choices:
        # rbf = 12 components
        # poly = 11 components
        # sigmoid = 9 components
        # cosine = 10 components
        
#------------ fit a KMeans clustering algorithm using the Kernel PCA components

n_components = 9 # obtained from model complexity plot (sigmoid kernel)
kpca = KernelPCA(n_components=n_components, kernel='sigmoid', random_state=42, n_jobs=-1)
kpca.fit(x_train) # fit model
x_etf_kpca = kpca.transform(x_train) 
kpca_experiment = experiment_kpca(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='etf',
                                n_components=n_components,# replaced with optimal components
                                name='etf_kpca_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=kpca_experiment[0], dataset='etf_kpca_reduced')

n_clusters = 4 # this is obtained from model complexity plot of kmeans
kmeans_kpca_clusters = kmeans_experiment(x_train=kpca_experiment[0], 
                                        x_test=kpca_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='etf_kpca_reduced')

#------------ fit a GMM clustering algorithm using the kernel PCA components

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=kpca_experiment[0], dataset='etf_kpca_reduced')

n_components = 18 # obtained from model complexity plot for GMM
gmm_kpca_clusters = gmm_experiment(x_train=kpca_experiment[0], 
                                   x_test=kpca_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='etf_kpca_reduced')



#%% 5. ETF DATASET (Randomized Projection REDUCED DATASET)


def create_viz(x_train, y, abs_diff, dataset, name):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    
    plt.subplot(131)
    df = pd.DataFrame(x_train[:, :2], columns=['gauss_prj1', 'gauss_prj2'])
    df['y'] = y_train
    colors = sns.color_palette('hls', len(np.unique(df['y'].dropna())))
    sns.scatterplot(x='gauss_prj1', y='gauss_prj2', hue='y', palette=colors, data=df, legend='full', alpha=0.3)
    # Annotate standard deviation arrows for the two components
    plt.annotate('', xy=(np.std(df['gauss_prj1']), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df['gauss_prj2'])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    # Set title and axes limits
    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df['gauss_prj1']))
    ylim = 1.1 * np.max(np.abs(df['gauss_prj2']))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    plt.subplot(132)
    plt.imshow(abs_diff)
    plt.colorbar()
    plt.title('Visualization of absolute differences')
    
    plt.subplot(133)
    ax = plt.hist(abs_diff.flatten())
    plt.title('Histogram of absolute differences')
    
    fig.subplots_adjust(wspace=.3)


#---------- plot model complexity

reduction_dim_gauss = []
eps_arr_gauss = []
mean_abs_diff_gauss = []
n_component_range = np.arange(5, 21,5)

for eps in np.arange(0.1, 0.999, 0.2):
    for n in n_component_range:
        #min_dim = johnson_lindenstrauss_min_dim(n_samples=x_train.shape[0], eps=eps)
        #if min_dim > x_train.shape[0]:
         #   continue
        gauss_proj = GaussianRandomProjection(n_components=n,random_state=42,eps=eps,)
        X_transform = gauss_proj.fit_transform(x_train)
        dist_raw = euclidean_distances(x_train)
        dist_transform = euclidean_distances(X_transform)
        abs_diff_gauss = abs(dist_raw - dist_transform) 
    
        create_viz(x_train=X_transform, y=y_train, abs_diff=abs_diff_gauss, dataset='churn', name='gauss_proj')
        plt.suptitle('eps = ' + '{:.2f}'.format(eps) + ', n_components = ' + str(X_transform.shape[1]))

        reduction_dim_gauss.append(100-X_transform.shape[1]/X_transform.shape[1]*100)
        eps_arr_gauss.append(eps)
        mean_abs_diff_gauss.append(np.mean(abs_diff_gauss.flatten()))
    

# ##################### #########   run rp several times and see what happens  ##############################
rand_gauss(x_train, dataset='etf')     
########################################################################

#------------ fit a KMeans clustering algorithm using the RGP components

n_components = 15 # obtained from model complexity plot for RGP
eps = 0.50
rgp = GaussianRandomProjection(n_components=n_components,random_state=42,eps=eps,)
x_etf_rgp = rgp.fit_transform(x_train)
rgp_experiment = experiment_rgp(x_train=x_train, y_train=y_train,
                                x_test=x_test, dataset='etf',
                                n_components=n_components,
                                eps=eps,
                                name='rgp_reduced')

# find performance using train data with dimensionality reduction

# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_KMeans(x=rgp_experiment[0], dataset='etf_rgp_reduced')

n_clusters = 3 # this is obtained from model complexity plot of kmeans
kmeans_ica_clusters = kmeans_experiment(x_train=rgp_experiment[0], 
                                        x_test=rgp_experiment[1], 
                                        y_train=y_train,
                                        y_test=y_test,
                                        n_clusters=n_clusters,
                                        dataset='etf_rgp_reduced')


#------------ fit a GMM clustering algorithm using the ICA components
# 1. check model complexity to determine the optimal n_clusters for kmeans
plot_model_complexity_GMM(x=rgp_experiment[0], dataset='etf_rgp_reduced')

n_components = 7 # obtained from model complexity plot for GMM
gmm_ica_clusters = gmm_experiment(x_train=rgp_experiment[0], 
                                   x_test=rgp_experiment[1],
                                   n_components=n_components,
                                   y_train=y_train, y_test=y_test,
                                   dataset='etf_rgp_reduced')




#%% NEURAL NETWORKS

# Grid search on original data
steps = [('Scaler', MinMaxScaler()), ('mlp', MLPClassifier(random_state=42))]
pipeline = Pipeline(steps)

parameters = {
    'mlp__hidden_layer_sizes': [(8, 10), (8, 8),
                                (10, 10), (50, 10,)],
    'mlp__activation': ['tanh', 'relu', 'logistic'],
    'mlp__solver': ['sgd', 'adam', 'lbfgs'],
    'mlp__alpha': [0.0001,0.001,0.01, 0.05],
    'mlp__learning_rate': ['constant','adaptive'],
    'mlp__early_stopping':[True, False] }

mlp_rcv = RandomizedSearchCV(estimator=pipeline,
                        param_distributions=parameters,
                        cv=5,
                        n_iter=10,
                        scoring='roc_auc',
                        n_jobs=-1)

# Training on and fetching the best parameters
mlp_rcv.fit(X_churn, y_churn)

mlp_rcv.best_params_

# {'mlp__solver': 'sgd',
#  'mlp__learning_rate': 'adaptive',
#  'mlp__hidden_layer_sizes': (10, 10),
#  'mlp__early_stopping': False,
#  'mlp__alpha': 0.001,
#  'mlp__activation': 'tanh'}


nn_experiment(x_train=x_train, x_test=x_test, y_train=y_train,y_test=y_test,
              x_pca=pca_experiment, x_kpca=kpca_experiment,
              x_rgp=rgp_experiment, x_ica=ica_experiment,
              kmeans_clusters_ica=kmeans_ica_clusters,
              kmeans_clusters_pca=kmeans_pca_clusters,
              kmeans_clusters_kpca=kmeans_kpca_clusters,
              kmeans_clusters_rgp=kmeans_rgp_clusters,
              gmm_clusters_ica=gmm_ica_clusters,
              gmm_clusters_pca=gmm_pca_clusters,
              gmm_clusters_kpca=gmm_kpca_clusters,
              gmm_clusters_rgp=gmm_rgp_clusters,
              learning_rate=0.05)




























