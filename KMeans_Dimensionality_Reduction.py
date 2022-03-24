#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:55:07 2022

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
    Unsupervised Learning and Dimensionality Reduction/images/'  # output images directory

#%% Data Preprocessing
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


#%%  ####################################          K - MEANS       ########################################

#%% 1. CHURN DATASET

# find a benchmark performance using all data without dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_churn.nunique(),)
clusters = model.fit_predict(X_churn, )
benchmark(X_churn, y_churn, clusters)

# Declare PCA and reduce data
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(X_churn)

# Declare TSNE and reduce data
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X_churn)


# Create dataframe for visualization
df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
df['pca1'] = x_pca[:, 0]
df['pca2'] = x_pca[:, 1]
df['y'] = y_churn
df['c'] = clusters
plot_components(component1='pca1', component2='pca2', df=df, name='PCA')
plot_components(component1='tsne1', component2='tsne2', df=df, name='tSNE')

# visualize clusters
viz_clusters_KMeans(X=X_churn, y=y_churn, dataset='Churn', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_churn, dataset='Churn')


#%% 2. CHURN DATASET (ICA REDUCED DATASET)
scaler = MinMaxScaler()

n_components = 2
model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(scaler.fit_transform(X_churn))

# fit the model to training data and compute MSE after reconstruction
x_reduced = model.transform(scaler.fit_transform(X_churn)) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_ica = np.mean((X_churn - x_reconstructed)**2)

# visualize components
df = pd.DataFrame(x_reduced[:, :2], columns=['component1','component2'])
df['y'] = y_churn


plot_components('component1', 'component2', df, dataset='churn', name='ica')


#---------- experiment
    
 def experiment_ica(x_train, y_train, dataset, name):
        """Perform experiments.
            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, Churn or ETF.
               name (string): name of dimensionality reduction technique.
            Returns:
              x_train_reduced (ndarray): reduced training data.
            """
        n_components = 2
        model = FastICA(n_components=n_components, random_state=42, max_iter=1000)

        print('\nTrain on training set')
        x_train_reduced = model.fit_transform(x_train)  # fit
        mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
        print('Reconstruction error:', mse)
        df = pd.DataFrame(x_train_reduced[:,:2], columns=['component1','component2'])
        df['y'] = y_train
        plot_components('component1', 'component2', df=df, dataset=dataset, name=name)  # visualize components

        return x_train_reduced  # return reduced training and test data

experiment_ica(x_train=X_churn, y_train=y_churn, dataset='churn', name='ica')


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


#------------ fit a KMeans clustering algorithm using the ICA components

n_components = 3 # obtained from model complexity plot

ica = FastICA(n_components=n_components, random_state=42,)
X_churn_ica = ica.fit_transform(X_churn)


# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_churn.nunique(),)
clusters = model.fit_predict(X_churn_ica, )
benchmark(X_churn_ica, y_churn, clusters)

# visualize clusters
viz_clusters_KMeans(X=ica.fit_transform(X_churn_ica), y=y_churn, dataset='Churn_ica_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=ica.fit_transform(X_churn_ica), dataset='Churn_ica_reduced')


#%% 3. CHURN DATASET (PCA REDUCED DATASET)

scaler = MinMaxScaler()
n_components = 2
model = PCA(n_components=n_components, random_state=42,)

# fit the model to training data and compute MSE after reconstruction
model.fit(scaler.fit_transform(X_churn))
x_reduced = model.transform(scaler.fit_transform(X_churn)) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_pca = np.mean((X_churn - x_reconstructed)**2)

# visualize components
df = pd.DataFrame(x_reduced[:, :2], columns=['component1','component2'])
df['y'] = y_churn


plot_components('component1', 'component2', df, dataset='churn', name='pca')


#---------- experiment
    
 def experiment_pca(x_train, y_train, dataset, name):
        """Perform experiments.
            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, Churn or ETF.
               name (string): name of dimensionality reduction technique.
            Returns:
              x_train_reduced (ndarray): reduced training data.
            """
        n_components = 2
        model = PCA(n_components=n_components, random_state=42,)

        print('\nTrain on training set')
        x_train_reduced = model.fit_transform(x_train)  # fit
        mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
        print('Reconstruction error:', mse)
        df = pd.DataFrame(x_train_reduced[:,:2], columns=['component1','component2'])
        df['y'] = y_train
        plot_components('component1', 'component2', df=df, dataset=dataset, name=name)  # visualize components

        return x_train_reduced  # return reduced training and test data

experiment_pca(x_train=X_churn, y_train=y_churn, dataset='churn', name='pca')


#---------- plot model complexity
scaler = MinMaxScaler()
pca = PCA(svd_solver='auto',n_components = 0.99, random_state=42) # 99% variance explaination
pca.fit(scaler.fit_transform(X_churn))

n_components = 2
# compute total explained variance by number of components
explained_variance = np.sum(pca.explained_variance_ratio_[:n_components])
print('Explained variance [ n compents = {}] = {:.3f}'.format(n_components, explained_variance))


# create plots
component_range = np.arange(1, len(pca.explained_variance_ratio_)+1)

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))

# plot the cumulative dsitribution of explained variance
ax1.plot(component_range, np.cumsum(pca.explained_variance_ratio_),marker='o',color='b', linestyle='--')
ax1.axhline(y=0.95, color='r', linestyle='-')
ax1.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
set_axis_title_labels(ax1, title='PCA: Choosing number of components (k) by explained variance ratio',
                      x_label='Number of components (k)', y_label='Cumulative variance (%)')
# plot explained variance ratio or eigenvalue distirbution
ax2.bar(component_range,pca.explained_variance_ratio_, color='cyan')
set_axis_title_labels(ax2, title='PCA - Eigenvalues Distribution', 
                      x_label='Number of components (k)', y_label='Variance (%)')

#------------ fit a KMeans clustering algorithm using the PCA components

n_components = 17 # obtained from model complexity plot

pca = PCA(n_components=17, random_state=42,)
pca.fit(scaler.fit_transform(X_churn))
X_churn_pca = pca.transform(scaler.fit_transform(X_churn)) # fit model

# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_churn.nunique(),)
clusters = model.fit_predict(X_churn_pca, )
benchmark(X_churn_pca, y_churn, clusters)

# visualize clusters
viz_clusters_KMeans(X=pca.fit_transform(X_churn_pca), y=y_churn, dataset='Churn_pca_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=pca.fit_transform(X_churn_pca), dataset='Churn_pca_reduced')




#%% 4. CHURN DATASET (Kernel PCA REDUCED DATASET)


#---------- experiment
    
 def experiment_kpca(x_train, y_train, dataset, name):
        """Perform experiments.
            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, Churn or ETF.
               name (string): name of dimensionality reduction technique.
            Returns:
              x_train_reduced (ndarray): reduced training data.
            """
        n_components = 2
        model = KernelPCA(n_components=n_components,kernel='rbf', random_state=42,fit_inverse_transform=True)

        print('\nTrain on training set')
        x_train_reduced = model.fit_transform(x_train)  # fit
        mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
        print('Reconstruction error:', mse)
        df = pd.DataFrame(x_train_reduced[:,:2], columns=['component1','component2'])
        df['y'] = y_train
        plot_components('component1', 'component2', df=df, dataset=dataset, name=name)  # visualize components

        return x_train_reduced  # return reduced training and test data

experiment_kpca(x_train=X_churn, y_train=y_churn, dataset='churn', name='kpca')


#---------- plot model complexity

scaler = MinMaxScaler()
n_components = 2
component_range = np.arange(1, X_churn.shape[1]+1)
kernels = ['rbf', 'sigmoid', 'poly', 'cosine']

# create plots
fig, ax = plt.subplots(4, 2, figsize=(15,30))
ax = ax.ravel()

for i, kernel in enumerate(kernels):
    
    kpca = KernelPCA(n_components=X_churn.shape[1], kernel=kernel, random_state=42, n_jobs=-1)
    kpca.fit(X_churn)
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


#------------ fit a KMeans clustering algorithm using the Kernel PCA components

n_components = 21 # obtained from model complexity plot

kpca = KernelPCA(n_components=n_components, kernel='rbf', random_state=42, n_jobs=-1)
kpca.fit(scaler.fit_transform(X_churn))
X_churn_kpca = kpca.transform(scaler.fit_transform(X_churn)) # fit model

# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_churn.nunique(),)
clusters = model.fit_predict(X_churn_kpca, )
benchmark(X_churn_kpca, y_churn, clusters)

# visualize clusters
viz_clusters_KMeans(X=X_churn_kpca, y=y_churn, dataset='Churn_kpca_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_churn_kpca, dataset='Churn_kpca_reduced')




# fit the model to training data and compute MSE after reconstruction
scaler = MinMaxScaler()
n_components = 21
model = KernelPCA(n_components=n_components, kernel='rbf', random_state=42, n_jobs=-1, fit_inverse_transform=True)

model.fit(scaler.fit_transform(X_churn))
x_reduced = model.transform(scaler.fit_transform(X_churn)) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_kpca = np.mean((X_churn - x_reconstructed)**2)




#%% 5. CHURN DATASET (Randomized Projection REDUCED DATASET)


def create_viz(X_transform, y, abs_diff, dataset, name):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    
    plt.subplot(131)
    df = pd.DataFrame(X_transform[:, :2], columns=['gauss_prj1', 'gauss_prj2'])
    df['y'] = y 
    colors = sns.color_palette('hls', len(np.unique(df['y'])))
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
    save_figure('{}_{}_components'.format(dataset, name))


#---------- plot model complexity

reduction_dim_gauss = []
eps_arr_gauss = []
mean_abs_diff_gauss = []

for eps in np.arange(0.01, 0.999, 0.02):

    #min_dim = johnson_lindenstrauss_min_dim(n_samples=X_churn.shape[0], eps=eps)
    #if min_dim > X_churn.shape[0]:
     #   continue
    gauss_proj = GaussianRandomProjection(n_components=2,random_state=42,eps=eps,)
    X_transform = gauss_proj.fit_transform(X_churn)
    dist_raw = euclidean_distances(X_churn)
    dist_transform = euclidean_distances(X_transform)
    abs_diff_gauss = abs(dist_raw - dist_transform) 

    create_viz(X_transform=X_transform, y=y_churn, abs_diff=abs_diff_gauss, dataset='churn', name='gauss_proj')
    plt.suptitle('eps = ' + '{:.2f}'.format(eps) + ', n_components = ' + str(X_transform.shape[1]))
    
    reduction_dim_gauss.append(100-X_transform.shape[1]/X_churn.shape[1]*100)
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



#%% Data Preprocessing

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
df.loc[df.retFut1 > df.retFut1[:split].quantile(q=0.66), 'Signal'] = 1 # we're not actually splitting the data here
df.loc[df.retFut1 < df.retFut1[:split].quantile(q=0.34), 'Signal'] = -1 # we're not actually splitting the data here

X_etf = df.drop(['Date','Open','Close','Adj Close','Signal', 'High','Low', 'Volume', 'retFut1'], axis=1)
y_etf = df['Signal']


# Original matrix to be used to calculate reconstruction error
X_ETF = df.drop(['Date','Open','Close','Adj Close','Signal', 'High','Low', 'Volume', 'retFut1'], axis=1)



#%% 1. ETF DATASET

scaler = MinMaxScaler()
X_etf = scaler.fit_transform(X_etf)

# find a benchmark performance using all data without dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_etf.nunique(),)
clusters = model.fit_predict(X_etf, )
benchmark(X_etf, y_etf, clusters)

# Declare PCA and reduce data
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(X_etf)

# Declare TSNE and reduce data
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X_etf)


# Create dataframe for visualization
df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
df['pca1'] = x_pca[:, 0]
df['pca2'] = x_pca[:, 1]
df['y'] = y_etf
df['c'] = clusters
plot_components(component1='pca1', component2='pca2', df=df, name='PCA', dataset='ETF')
plot_components(component1='tsne1', component2='tsne2', df=df, name='tSNE', dataset='ETF')

# visualize clusters
viz_clusters_KMeans(X=X_etf, y=y_etf, dataset='ETF', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_etf, dataset='ETF')


#%% 2. ETF DATASET (ICA REDUCED DATASET)
scaler = MinMaxScaler()

n_components = 2
model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(scaler.fit_transform(X_etf))

# fit the model to training data and compute MSE after reconstruction
x_reduced = model.transform(X_etf) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_ica = np.mean((X_etf - x_reconstructed)**2)

# visualize components
df = pd.DataFrame(x_reduced[:, :2], columns=['ica1','ica2'])
df['y'] = y_etf


plot_components('ica1', 'ica2', df, dataset='ETF', name='ica')


#---------- experiment
   
def experiment_ica(x_train, y_train, dataset, name):
       """Perform experiments.
           Args:
              x_train (ndarray): training data.
              x_test (ndarray): test data.
              y_train (ndarray): training labels.
              dataset (string): dataset, Churn or ETF.
              name (string): name of dimensionality reduction technique.
           Returns:
             x_train_reduced (ndarray): reduced training data.
           """
       n_components = 2
       model = FastICA(n_components=n_components, random_state=42, max_iter=1000)

       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['ica1','ica2'])
       df['y'] = y_train
       plot_components('ica1', 'ica2', df=df, dataset=dataset, name=name)  # visualize components

       return x_train_reduced  # return reduced training and test data

experiment_ica(x_train=X_etf, y_train=y_etf, dataset='ETF', name='ica')


#---------- plot model complexity

component_range = np.arange(2, X_etf.shape[1]+1)

avg_kurt = [] 
for component in component_range:
    
    ica  = FastICA(n_components=component, max_iter=1000, random_state=42)
    ica.fit(X_etf)
    component_kurt = kurtosis(ica.components_, axis=1, fisher=False)
    avg_kurt.append(np.mean(component_kurt))
    print('k = {} --> average kurtosis = {:.3f}'.format(component, avg_kurt[-1]))
    
# create kurtosis plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10))
ax1.plot(component_range, avg_kurt, '-.', markersize=1, label='kurtosis')
set_axis_title_labels(ax1, title='ICA: Choosing number of components (k) by average kurtosis',
                      x_label='Number of Components (k)', y_label='Average Kurtosis')

n_components =  X_etf.shape[1]

model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
model.fit(X_etf)  # Fit ICA on churn data
# plot axis
x_range = np.arange(1, n_components + 1)
component_kurt = kurtosis(model.components_, axis=1, fisher=False)
ax2.bar(x_range, component_kurt, color='cyan')
ax2.set_xticks(x_range)
set_axis_title_labels(ax2, title='ICA: Kurtosis Distribution of Components',
                      x_label='Independent Component (k)',
                      y_label='Kurtosis')
plt.savefig('images/etf_ica_model_complexity.png')


#------------ fit a KMeans clustering algorithm using the ICA components

n_components = 10 # obtained from model complexity plot

ica = FastICA(n_components=n_components, random_state=42,)
X_etf_ica = ica.fit_transform(X_etf)


# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_etf.nunique(),)
clusters = model.fit_predict(X_etf_ica, )
benchmark(X_etf_ica, y_etf, clusters)

# visualize clusters
viz_clusters_KMeans(X=X_etf_ica, y=y_etf, dataset='ETF_ica_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_etf_ica, dataset='ETF_ica_reduced')


#------------- fit the model to training data and compute MSE after reconstruction
n_components = 10
model = FastICA(n_components=n_components, random_state=42,)

model.fit(X_etf)
x_reduced = model.transform(X_etf) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_ica = np.mean((X_etf- x_reconstructed)**2)























#%% 3. ETF DATASET (PCA REDUCED DATASET)

n_components = 2
model = PCA(n_components=n_components, random_state=42,)

# fit the model to training data and compute MSE after reconstruction
model.fit(scaler.fit_transform(X_etf))
x_reduced = model.transform(scaler.fit_transform(X_etf)) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_pca = np.mean((X_etf - x_reconstructed)**2)

# visualize components
df = pd.DataFrame(x_reduced[:, :2], columns=['pca1','pca2'])
df['y'] = y_etf


plot_components('pca1', 'pca2', df, dataset='ETF', name='pca')


#---------- experiment
   
def experiment_pca(x_train, y_train, dataset, name):
       """Perform experiments.
           Args:
              x_train (ndarray): training data.
              x_test (ndarray): test data.
              y_train (ndarray): training labels.
              dataset (string): dataset, Churn or ETF.
              name (string): name of dimensionality reduction technique.
           Returns:
             x_train_reduced (ndarray): reduced training data.
           """
       n_components = 2
       model = PCA(n_components=n_components, random_state=42,)

       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['pca1','pca2'])
       df['y'] = y_train
       plot_components('pca1', 'pca2', df=df, dataset=dataset, name=name)  # visualize components

       return x_train_reduced  # return reduced training and test data

experiment_pca(x_train=X_etf, y_train=y_etf, dataset='ETF', name='pca')


#---------- plot model complexity
pca = PCA(svd_solver='auto',n_components = 0.99, random_state=42) # 99% variance explaination
pca.fit(scaler.fit_transform(X_etf))

n_components = 2
# compute total explained variance by number of components
explained_variance = np.sum(pca.explained_variance_ratio_[:n_components])
print('Explained variance [ n compents = {}] = {:.3f}'.format(n_components, explained_variance))


# create plots
component_range = np.arange(1, len(pca.explained_variance_ratio_)+1)

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))

# plot the cumulative dsitribution of explained variance
ax1.plot(component_range, np.cumsum(pca.explained_variance_ratio_),marker='o',color='b', linestyle='--')
ax1.axhline(y=0.95, color='r', linestyle='-')
ax1.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
set_axis_title_labels(ax1, title='PCA: Choosing number of components (k) by explained variance ratio',
                      x_label='Number of components (k)', y_label='Cumulative variance (%)')
# plot explained variance ratio or eigenvalue distirbution
ax2.bar(component_range,pca.explained_variance_ratio_, color='cyan')
set_axis_title_labels(ax2, title='PCA - Eigenvalues Distribution', 
                      x_label='Number of components (k)', y_label='Variance (%)')

#------------ fit a KMeans clustering algorithm using the PCA components

n_components = 9 # obtained from model complexity plot

pca = PCA(n_components=n_components, random_state=42,)
pca.fit(X_etf)
X_etf_pca = pca.transform(X_etf) # fit model

# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_etf.nunique(),)
clusters = model.fit_predict(X_etf_pca, )
benchmark(X_etf_pca, y_etf, clusters)

# visualize clusters
viz_clusters_KMeans(X=X_etf_pca, y=y_etf, dataset='ETF_pca_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_etf_pca, dataset='ETF_pca_reduced')

#------------- fit the model to training data and compute MSE after reconstruction
n_components = 9
model = PCA(n_components=n_components, random_state=42,)

model.fit(X_etf)
x_reduced = model.transform(X_etf) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_pca = np.mean((X_etf- x_reconstructed)**2)



#%% 4. ETF DATASET (Kernel PCA REDUCED DATASET)


#---------- experiment
    
def experiment_kpca(x_train, y_train, dataset, name):
       """Perform experiments.
           Args:
              x_train (ndarray): training data.
              x_test (ndarray): test data.
              y_train (ndarray): training labels.
              dataset (string): dataset, Churn or ETF.
              name (string): name of dimensionality reduction technique.
           Returns:
             x_train_reduced (ndarray): reduced training data.
           """
       n_components = 2
       model = KernelPCA(n_components=n_components,kernel='rbf', random_state=42,fit_inverse_transform=True)

       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['kpca1','kpca2'])
       df['y'] = y_train
       plot_components('kpca1', 'kpca2', df=df, dataset=dataset, name=name)  # visualize components

       return x_train_reduced  # return reduced training and test data

experiment_kpca(x_train=X_etf, y_train=y_etf, dataset='ETF', name='kpca')


#---------- plot model complexity

n_components = 2 
component_range = np.arange(1, X_etf.shape[1]+1)
kernels = ['rbf', 'sigmoid', 'poly', 'cosine']

# create plots
fig, ax = plt.subplots(4, 2, figsize=(15,30))
ax = ax.ravel()


for i, kernel in enumerate(kernels):
    
    kpca = KernelPCA(n_components=X_etf.shape[1], kernel=kernel, random_state=42, n_jobs=-1)
    kpca.fit(X_etf)
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

save_figure_tight('{}_kpca_model_complexity'.format('ETF'))


#------------ fit a KMeans clustering algorithm using the PCA components

n_components = 10 # obtained from model complexity plot

kpca = KernelPCA(n_components=n_components, kernel='rbf', random_state=42, n_jobs=-1)
kpca.fit(X_etf)
X_etf_kpca = kpca.transform(X_etf) # fit model

# find a benchmark performance using all data with dimensionality reduction
model = KMeans(random_state=42, n_clusters = y_etf.nunique(),)
clusters = model.fit_predict(X_etf_kpca, )
benchmark(X_etf_kpca, y_etf, clusters)

# visualize clusters
viz_clusters_KMeans(X=X_etf_kpca, y=y_etf, dataset='ETF_kpca_reduced', method='kmeans')

# visualize model complexity 
plot_model_complexity_KMeans(x=X_etf_kpca, dataset='ETF_kpca_reduced')


#------------- fit the model to training data and compute MSE after reconstruction
n_components = 10
model = KernelPCA(n_components=n_components, kernel='rbf', random_state=42, n_jobs=-1, fit_inverse_transform=True)

model.fit(X_etf)
x_reduced = model.transform(X_etf) # fit model
 
# reconstruct original dataset and compute MSE
x_reconstructed = model.inverse_transform(x_reduced)
mse_kpca = np.mean((X_etf- x_reconstructed)**2)
































#%% 5. ETF DATASET (Randomized Projection REDUCED DATASET)


def create_viz(X_transform, y, abs_diff, dataset, name):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    
    plt.subplot(131)
    df = pd.DataFrame(X_transform[:, :2], columns=['gauss_prj1', 'gauss_prj2'])
    df['y'] = y 
    colors = sns.color_palette('hls', len(np.unique(df['y'])))
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
    save_figure('{}_{}_components'.format(dataset, name))


#---------- plot model complexity

reduction_dim_gauss = []
eps_arr_gauss = []
mean_abs_diff_gauss = []

for eps in np.arange(0.01, 0.999, 0.02):

    #min_dim = johnson_lindenstrauss_min_dim(n_samples=X_etf.shape[1], eps=eps)
    #if min_dim > X_etf.shape[1]:
        #continue
    gauss_proj = GaussianRandomProjection(n_components=2,random_state=42,eps=eps,)
    X_transform = gauss_proj.fit_transform(X_etf)
    dist_raw = euclidean_distances(X_etf)
    dist_transform = euclidean_distances(X_transform)
    abs_diff_gauss = abs(dist_raw - dist_transform) 

    create_viz(X_transform=X_transform, y=y_etf, abs_diff=abs_diff_gauss, dataset='churn', name='gauss_proj')
    plt.suptitle('eps = ' + '{:.2f}'.format(eps) + ', n_components = ' + str(X_transform.shape[1]))
    
    reduction_dim_gauss.append(100-X_transform.shape[1]/X_churn.shape[1]*100)
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

