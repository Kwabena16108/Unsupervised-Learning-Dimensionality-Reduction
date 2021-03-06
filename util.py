#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:03:30 2022

@author: dicksonnkwantabisa
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, \
                            homogeneity_score, v_measure_score, silhouette_samples, silhouette_score

import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier



IMAGE_DIR = '/Users/dicksonnkwantabisa/Desktop/CS7641-MachineLearning/Unsupervised Learning and Dimensionality Reduction/images/'  # output images directory


def evaluate_kmeans(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    # silhoutte score and elbow score
    return silhouette_score(X, clusters), kmeans.inertia_


def kmeans_report(k_vals, X):
    s_scores, e_scores = [], [] # silhoutte score and elbow score
    for k in tqdm(k_vals):
        s,e = evaluate_kmeans(k, X)
        s_scores.append(s)
        e_scores.append(e)
    
    plt.figure(figsize=(10, 10))
    sns.barplot(x=k_vals, y=s_scores)
    plt.xlabel("Clusters")
    plt.ylabel("Silhoutte Score")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    sns.lineplot(x=k_vals, y=e_scores)
    plt.xlabel("K")
    plt.ylabel("Sum of Squares")
    plt.show()
    
    
def KMeans_Report(X,range_n_clusters):    
    for clusters in range_n_clusters:
        # create a subplot
        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(10, 6)
        # silhouette plot
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (clusters+1) * 10])

        cl = KMeans(n_clusters=clusters, random_state=42)
        cl_labels = cl.fit_predict(X)

        silhouette_avg = silhouette_score(X, cl_labels)
        print(
        "For n_clusters=", clusters,
        "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cl_labels)

        y_lower = 10
        for i in range(clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cl_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = {}".format(clusters),
            fontsize=14, fontweight="bold"
        )



    plt.show()
    
 




def benchmark(x, y, clusters):
        """Benchmark the model.
           Args:
               x (ndarray): data.
               y (ndarray): true labels.
               clusters (ndarray): clusters found by the model.
           Returns:
               None.
           """
        print('homo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
        print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(homogeneity_score(y, clusters),
                                                                      completeness_score(y, clusters),
                                                                      v_measure_score(y, clusters),
                                                                      adjusted_rand_score(y, clusters),
                                                                      adjusted_mutual_info_score(y, clusters, average_method='arithmetic'),
                                                                      silhouette_score(x, clusters, metric='euclidean')))

def plot_clusters(ax, component1, component2, df, name):
    """Plot clusters for clustering.
        Args:
          ax (axes object): axes to plot at.
          component1 (string): first component to plot (PCA or TSNE).
          component2 (string): second component to plot (PCA or TSNE).
          df (dataframe): dataframe containing input data.
          name (string): clustering technique.
        Returns:
          None.
        """

    # Plot input data onto first two components at given axes
    y_colors = sns.color_palette('hls', len(np.unique(df['y'].dropna())))
    c_colors = sns.color_palette('hls', len(np.unique(df['c'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax[0])
    sns.scatterplot(x=component1, y=component2, hue='c', palette=c_colors, data=df, legend='full', alpha=0.3, ax=ax[1])

    # Set titles
    ax[0].set_title('True Clusters represented with {}'.format(component1[:-1].upper()))
    ax[1].set_title('{} Clusters represented with {}'.format(name.upper(), component1[:-1].upper()))

    # Set axes limits
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)

def plot_components(component1, component2, df, dataset,name):
    """Plot components for dimensionality reduction.
        Args:
          component1 (string): first component to plot.
          component2 (string): second component to plot.
          df (dataframe): dataframe containing input data.
          method (string): dimensionality reduction technique.
          dataset (string): name of dataset
        Returns:
          None.
        """
    # Create figure and plot input data onto first two components
    plt.figure()
    colors = sns.color_palette('hls', len(np.unique(df['y'].dropna())))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=colors, data=df, legend='full', alpha=0.3)

    # Annotate standard deviation arrows for the two components
    plt.annotate('', xy=(np.std(df[component1]), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df[component2])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))

    # Set title and axes limits
    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    save_figure('{}_{}_components'.format(dataset, name))
    
def split_dataset(X, y, dataset):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42,
        test_size = 0.2)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)    
    print('\nTotal  data size of {} :'.format(dataset))
    print('Training data set: {}'.format(x_train.shape))
    print('Testing data set: {}'.format(x_test.shape))
    return x_train, x_test, y_train, y_test


def time_series_data_split(X, y, dataset):
    t = .8
    split = int(t*len(X))
    x_train, x_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)    
    print('\nTotal  data size of {} :'.format(dataset))
    print('Training data set: {}'.format(x_train.shape))
    print('Testing data set: {}'.format(x_test.shape))
    return x_train, x_test, y_train, y_test


def kmeans_experiment(x_train, x_test, y_train, y_test,dataset,n_clusters=2):
    # plot model complexity
    plot_model_complexity_KMeans(x_train, dataset)
    # fit the model and benchmark on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_train = kmeans.fit_predict(x_train)
    clusters_test = kmeans.predict(x_test)
    print('\nTraining score:')
    benchmark(x_train, y_train, clusters_train)
    print('\nTesting score:')
    benchmark(x_test, y_test, clusters_test)
    # ##  plot clusters # ##
    # Declare PCA and reduce data
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_train)
    # Declare TSNE and reduce data
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_train)
    # Create dataframe for visualization
    df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
    df['pca1'] = x_pca[:, 0]
    df['pca2'] = x_pca[:, 1]
    df['y'] = y_train
    df['c'] = clusters_train
    # Create subplot and plot clusters with PCA and TSNE
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 10))
    plot_clusters(ax1, 'pca1', 'pca2', df, name='PCA')
    plot_clusters(ax2, 'tsne1', 'tsne2', df, name='tSNE')
    # Save figure
    save_figure_tight('{}_{}_clusters'.format(dataset,'kmeans'))
    return clusters_train, clusters_test
    
def gmm_experiment(x_train, x_test, y_train, y_test,dataset,n_components=2):
    # plot model complexity
    plot_model_complexity_GMM(x_train, dataset)
    # fit the model and benchmark on training data
    gmm = GaussianMixture(n_components=n_components,
                          random_state=42, covariance_type='full',
                          max_iter=1000, n_init=10, init_params='random',)
    clusters_train = gmm.fit_predict(x_train)
    clusters_test = gmm.predict(x_test)
    print('\nTraining scores:')
    benchmark(x_train, y_train, clusters_train)
    print('\nTesting scores:')
    benchmark(x_test, y_test, clusters_test)
    # ##  plot clusters # ##
    # Declare PCA and reduce data
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_train)
    # Declare TSNE and reduce data
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_train)
    # Create dataframe for visualization
    df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
    df['pca1'] = x_pca[:, 0]
    df['pca2'] = x_pca[:, 1]
    df['y'] = y_train
    df['c'] = clusters_train
    # Create subplot and plot clusters with PCA and TSNE
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 10))
    plot_clusters(ax1, 'pca1', 'pca2', df, name='PCA')
    plot_clusters(ax2, 'tsne1', 'tsne2', df, name='tSNE')
    # Save figure
    save_figure_tight('{}_{}_clusters'.format(dataset,'gmm'))
    return clusters_train, clusters_test
   
def experiment_ica(x_train, y_train,x_test, dataset,n_components, name):
       """Perform experiments.
           Args:
              x_train (ndarray): training data.
              x_test (ndarray): test data.
              y_train (ndarray): training labels.
              dataset (string): dataset, Churn or ETF.
              n_components: number of components
              name (string): name of dimensionality reduction technique.
           Returns:
             x_train_reduced (ndarray): reduced training data.
           """
       model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('ICA reconstruction error is :', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['ica1','ica2'])
       df['y'] = y_train
       plot_components('ica1', 'ica2', df=df, dataset=dataset, name=name)  # visualize components
       return x_train_reduced, model.transform(x_test), model.inverse_transform(x_train_reduced)  # return reduced training and test data

    
def experiment_pca(x_train, y_train,x_test, dataset,n_components, name):
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
       model = PCA(n_components=n_components, random_state=42)

       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['ica1','ica2'])
       df['y'] = y_train
       plot_components('ica1', 'ica2', df=df, dataset=dataset, name=name)  # visualize components
       return x_train_reduced, model.transform(x_test),model.inverse_transform(x_train_reduced)  # return reduced training and test data    
    
def experiment_kpca(x_train, y_train,x_test, dataset,n_components, name):
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
       model = KernelPCA(n_components=n_components, kernel='sigmoid',
                         fit_inverse_transform=True,random_state=42, n_jobs=-1)
       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2)
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['ica1','ica2'])
       df['y'] = y_train
       plot_components('ica1', 'ica2', df=df, dataset=dataset, name=name)  # visualize components
       return x_train_reduced, model.transform(x_test),model.inverse_transform(x_train_reduced)  # return reduced training and test data    
   
def experiment_rgp(x_train, y_train,x_test, dataset,n_components,eps, name):
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
       model = InvertibleRandomProjection(n_components=n_components,random_state=42,eps=eps,)
       print('\nTrain on training set')
       x_train_reduced = model.fit_transform(x_train)  # fit
       mse = np.mean((x_train - model.inverse_transform(x_train_reduced))**2) # pseudo inverse
       print('Reconstruction error:', mse)
       df = pd.DataFrame(x_train_reduced[:,:2], columns=['ica1','ica2'])
       df['y'] = y_train
       plot_components('ica1', 'ica2', df=df, dataset=dataset, name=name)  # visualize components
       return x_train_reduced, model.transform(x_test),model.inverse_transform(x_train_reduced)  # return reduced training and test data    
  


def rand_gauss(x, dataset):
    """
    run several random gaussian projections
    and see what happen!
    LOL !!
    """
# Authors: Alexandre Gramfort, Gael Varoquaux
# License: BSD 3 clause
# #############################################################################
    for component in range(1,5):
        rgp = InvertibleRandomProjection(n_components=component,)
        
        runs =1000
        rgp_dim = np.zeros((x.shape[0], component, runs))
        for run in range(runs):
            rgp = InvertibleRandomProjection(n_components=component,random_state=np.random.randint(0,16108),)
            x_transform = rgp.fit_transform(x)
            rgp_dim[:,:, run]+=x_transform
        run_average = np.mean(rgp_dim, axis=2)
        run_st_dev = np.std(rgp_dim, axis=2)
        component_average = np.mean(run_average, axis=0)
        component_st_dev = np.std(run_st_dev, axis=0)
    
    # plot distribution
    plt.hist([np.random.normal(loc=component_average[idx], scale=component_st_dev[idx], size=10000) for idx in range(len(component_st_dev))],
             bins=100, histtype='step', label=['RP_1', 'RP_2', 'RP_3', 'RP_4']);
    [plt.plot([component_average[idx],component_average[idx]],[0,400],label='RP_'+str(idx)+'_avg',linestyle='dashed') for idx in range(len(component_average))]
    plt.legend()
    plt.title('Distribution of 1000 runs of Radomized Gaussian Projections')
    plt.ylabel('Value Count (100k sample)')
    plt.xlabel('Random Component value')
    plt.tight_layout()
    plt.show()
    save_figure_tight('{}_RP_histograms'.format(dataset))          
        
  
def viz_clusters_KMeans(X, y, dataset, method):
    # Declare PCA and reduce data
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(X)
    
    # Declare TSNE and reduce data
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(X)
    
    n_classes = len(np.unique(y)) # number of clusters
    print('\nBenchmark Model with k = n classes = {}'.format(n_classes))
    
     # Benchamark the model with number of clusters (k) = number of classes
    name_param = "n_clusters"
    model = KMeans(random_state=42)
    model_params = model.get_params()
    model_params[name_param] = n_classes
    model.set_params(**model_params)
    clusters = model.fit_predict(X)
    
    # Create dataframe for visualization
    df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
    df['pca1'] = x_pca[:, 0]
    df['pca2'] = x_pca[:, 1]
    df['y'] = y
    df['c'] = clusters
    
    # Create subplot and plot clusters with PCA and TSNE
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 9))
    plot_clusters(ax1, 'pca1', 'pca2', df, name='PCA')
    plot_clusters(ax2, 'tsne1', 'tsne2', df, name='tSNE')
    
    # Save figure
    save_figure_tight('{}_{}_clusters'.format(dataset, method))

def viz_clusters_GMM(X, y, dataset, method):
    # Declare PCA and reduce data
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(X)
    
    # Declare TSNE and reduce data
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(X)
    
    n_components = len(np.unique(y.dropna())) # number of clusters
    print('\nBenchmark Model with k = n classes = {}'.format(n_components))
    
     # Benchamark the model with number of clusters (k) = number of classes
    name_param = "n_components"
    model = GaussianMixture(random_state=42, 
                            covariance_type='full',
                            max_iter=1000, n_init=10, 
                            init_params='random',)
    model_params = model.get_params()
    model_params[name_param] = n_components
    model.set_params(**model_params)
    clusters = model.fit_predict(X)
    
    # Create dataframe for visualization
    df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
    df['pca1'] = x_pca[:, 0]
    df['pca2'] = x_pca[:, 1]
    df['y'] = y
    df['c'] = clusters
    
    # Create subplot and plot clusters with PCA and TSNE
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13, 9))
    plot_clusters(ax1, 'pca1', 'pca2', df, name='PCA')
    plot_clusters(ax2, 'tsne1', 'tsne2', df, name='tSNE')
    
    # Save figure
    save_figure_tight('{}_{}_clusters'.format(dataset, method))



def plot_model_complexity_KMeans(x, dataset):
       """Perform and plot model complexity.
           Args:
              x (ndarray): training data.
              dataset (string): dataset, Churn or ETF.
           Returns:
              None.
           """
       
       print('\nPlot Model Complexity')
       max_n_clusters = 6
       inertia, inertia_diff = [], []  # inertia and delta inertia
       k_range = np.arange(1, max_n_clusters + 2)  # range of number of clusters k to plot over

       # For each k in the range
       for k in k_range:

           # Define a new k-Means model, fit on training data and report inertia
           model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=42,)
           model.fit(x)
           inertia.append(model.inertia_)
           print('k = {} -->  inertia = {:.3f}'.format(k, inertia[-1]))

           # Except for k=1, report also the delta of inertia
           if k > 1:
               inertia_diff.append(abs(inertia[-1] - inertia[-2]))

       # Create subplots and plot inertia and delta inertia on the first suplot
       fig, ax = plt.subplots(2, math.ceil(max_n_clusters / 2), figsize=(16, 10))
       ax = ax.flatten()
       ax[0].plot(k_range, inertia, '-o', markersize=2, label='Inertia')
       ax[0].plot(k_range[1:], inertia_diff, '-o', markersize=2, label=r'Inertia |$\Delta$|')

       # Set legend, title and labels
       ax[0].legend(loc='best')
       set_axis_title_labels(ax[0], title='K-MEANS - Choosing k with the Elbow method',
                                   x_label='Number of clusters k', y_label='Inertia')

       # Plot silhouette for the different k values
       visualize_silhouette(x, ax)

       # Save figure
       save_figure_tight('{}_kmeans_model_complexity'.format(dataset))



def plot_model_complexity_GMM(x, dataset,n_components_range = range(1, 21)):  
    """Perform and plot model complexity.
        Args:
           x (ndarray): training data.
           dataset (string): dataset, Churn or ETF.
        Returns:
           None.
        """
    
    print('\nPlot Model Complexity')
    aic, bic = [], []  # AIC and BIC scores
    lowest_bic = np.infty
    # n_components_range = range(1, 11)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    for cv_type in cv_types:
        # For each k in the range
        for n_components in n_components_range:
            # Define a new gmm model, fit on training data and report inertia
            model = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            model.fit(x)
            aic.append(model.aic(x))
            bic.append(model.bic(x))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = model
                
            print('with cv = {}, k = {} -->  aic = {:.3f}, bic = {:.3f}'.format(cv_type, 
                                                                                n_components,
                                                                                aic[-1],bic[-1]))
                                                                                
                                                                                
    
    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []
    # plot only BIC
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)
    
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(x)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        v, w = np.linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(x[Y_ == i, 0], x[Y_ == i, 1], 0.8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    
    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {best_gmm.covariance_type} model, "
        f"{best_gmm.n_components} components")
    
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
 
    # Save figure
    save_figure_tight('{}_gmm_model_complexity'.format(dataset))



def visualize_silhouette(x, ax):
    """Visualize silhouette.
        Args:
           x (ndarray): training data.
           ax (ndarray): vector of axes to plot at.
        Returns:
           None.
        """

    # For all k values starting from k=2 (for k=1, the silhouette score is not defined)
    max_n_clusters = 6
    for k in range(2, max_n_clusters + 1):

        # Define a new k-Means model, fit on training data and report clusters and  average silhouette
        model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=42,)
        clusters = model.fit_predict(x)
        silhouette_avg = silhouette_score(x, clusters, metric='euclidean')
        print('k = {} -->  average silhouette = '.format(k), silhouette_avg)

        # Plot red dashed vertical line for average silhouette
        ax[k-1].axvline(x=silhouette_avg, color="red", linestyle="--")

        # Compute silhouette scores and maximum silhouette
        silhouette = silhouette_samples(x, clusters, metric='euclidean')
        max_silhouette = np.max(silhouette)

        y_lower = 10  # starting y for plotting silhouette

        # For each cluster found
        for i in range(k):

            # Sort silhouette of current cluster
            silhouette_cluster = silhouette[clusters == i]
            silhouette_cluster.sort()

            # Compute the upper y for plotting silhouette
            y_upper = y_lower + silhouette_cluster.shape[0]

            # Fill the area corresponding to the current cluster silhouette scores
            color = plt.cm.nipy_spectral(float(i) / k)
            ax[k-1].fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_cluster,
                                  facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax[k-1].text(-0.1, y_lower + 0.5 * silhouette_cluster.shape[0], str(i))

            # Compute the new lower y for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # Set title and labels
        set_axis_title_labels(ax[k-1], title='K-MEANS - Silhouette for k = {}'.format(k),
                                    x_label='Silhouette', y_label='Silhouette distribution per Cluster')

        # Clear the y axis labels and set the x ones
        ax[k-1].set_yticks([])
        ax[k-1].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Set x and y limits
        ax[k-1].set_xlim(-0.2, 0.1 + round(max_silhouette, 1))
        ax[k-1].set_ylim(0, len(x) + (k + 1) * 10)
     
        
def set_axis_title_labels(ax, title, x_label, y_label):
    """Set axis title and labels.
        Args:
          ax (axis object): axis.
          title (string): plot title.
          x_label (string): x label.
          y_label (string): y label.
        Returns:
          None.
        """
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def set_plot_title_labels(title, x_label, y_label):
    """Set plot title and labels.
        Args:
          title (string): plot title.
          x_label (string): x label.
          y_label (string): y label.
        Returns:
          None.
        """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')


def save_figure_tight(title):
    """Save Figure with tight layout.
        Args:
          title (string): plot title.
        Returns:
          None.
        """
    plt.tight_layout()
    save_figure(title)


def save_figure(title):
    """Save Figure.
        Args:
          title (string): plot title.
        Returns:
          None.
        """
    plt.savefig(IMAGE_DIR + title)
    plt.close()
        
   
class InvertibleRandomProjection(GaussianRandomProjection):
    """Gaussian random projection with an inverse transform using the pseudoinverse."""

    def __init__(
        self, n_components="auto", eps=0.3, orthogonalize=False, random_state=None
    ):
        self.orthogonalize = orthogonalize
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)

    @property
    def pseudoinverse(self):
        """Pseudoinverse of the random projection.

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
        try:
            return self._pseudoinverse
        except AttributeError:
            if self.orthogonalize:
                # orthogonal matrix: inverse is just its transpose
                self._pseudoinverse = self.components_
            else:
                self._pseudoinverse = np.linalg.pinv(self.components_.T)
            return self._pseudoinverse
    def fit(self, X):
         super().fit(X)
         if self.orthogonalize:
             Q, _ = np.linalg.qr(self.components_.T)
             self.components_ = Q.T
         return self
     
    def inverse_transform(self, X):
        return X.dot(self.pseudoinverse)
         

   
  #%% The following functions/classes are used in the Neural Networks

class NeuralNetwork:

    def __init__(self, layer1_nodes, layer2_nodes, learning_rate):
        """Initialize a Neural Network classifier.
            Args:
                layer1_nodes (int): number of neurons in the first layer.
                layer2_nodes (int): number of neurons in the second layer.
                learning_rate (float): learning rate.
            Returns:
                None.
            """
        self.model = MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation='logistic',
                                   solver='sgd', alpha=0.001, batch_size=200, learning_rate='adaptive',
                                   learning_rate_init=learning_rate, max_iter=100, tol=1e-4,
                                   early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                   n_iter_no_change=100, random_state=42)

    def evaluate(self, x_test, y_test,x_train,y_train):
        """Evaluate the model by reporting the classification report
        the confusion matrix and roc scores.
            Args:
                x_test (ndarray): test data.
                y_test (ndarray): test labels.
            Returns:
                None.
            """
        predictions = self.predict(x_test)  # predict on test data
        y_score = cross_val_predict(self.model, X=x_train, y=y_train,
                                    cv=5, method='predict_proba')[:,1]
        pred_scores = dict(y_true=y_train, y_score=y_score)
        
        print('\nEvaluate on the Test Set')
        print(classification_report(y_test, predictions))  # produce classification report
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, predictions))  # produce confusion matrix
        print('AUC_ROC score:')
        print(roc_auc_score(**pred_scores))  # produce roc score


    def fit(self, x_train, y_train):
        """Fit the model by on the training data.
            Args:
                x_train (ndarray): training data.
                y_train (ndarray): training labels.
            Returns:
                None.
            """
        # Fit the model and report training time
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()

        print('\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, x):
        """Predict on some data.
            Args:
                x (ndarray): data to predict.
            Returns:
                predictions (ndarray): array of labels predictions.
            """
        # Predict and report inference time
        start_time = time.time()
        predictions = self.model.predict(x)
        end_time = time.time()

        print('\nPredicting on Testing Set: {:.4f} seconds'.format(end_time-start_time))

        return predictions

    def experiment(self, x_train, x_test, y_train, y_test):
        """Run an experiment on the model.
            Fit on training data and evaluate the model on the test set.
            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               y_test (ndarray): test labels.
            Returns:
               None.
            """
        self.fit(x_train, y_train)
        self.evaluate(x_test, y_test, x_train, y_train)
        
              

def nn_experiment(x_train, x_test, y_train, y_test,
                  x_ica, x_pca, x_kpca, x_rgp,
                  kmeans_clusters_ica, kmeans_clusters_pca,
                  kmeans_clusters_kpca, kmeans_clusters_rgp,
                  gmm_clusters_ica, gmm_clusters_pca,
                  gmm_clusters_kpca, gmm_clusters_rgp,
                  **kwargs):
    """
    this function will run neural net experiments
    1. run NN on each reduced data ( NN + ICA,PCA,kPCA,RGP)
    2. add clustring results each from KMeans and GMM to 
    original features and rerun NN
    """
    
    print('\n------------')
    print('NN')
    print('---------------')
    
    # run NN on original dataset
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    nn.experiment(x_train, x_test, y_train, y_test)
    
    
    print('\n------------')
    print('NN + PCA')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    nn.experiment(x_pca[0], x_pca[1], y_train, y_test)

    print('\n------------')
    print('NN + kPCA')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    nn.experiment(x_kpca[0], x_kpca[1], y_train, y_test)
    
    print('\n------------')
    print('NN + ICA')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    nn.experiment(x_ica[0], x_ica[1], y_train, y_test)
    
    
    print('\n------------')
    print('NN + RGP')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    nn.experiment(x_rgp[0], x_rgp[1], y_train, y_test)
    
    print('\n------------')
    print('NN + kmeans_ica_clusters + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    ica_km_normed = (kmeans_clusters_ica[0]-np.min(kmeans_clusters_ica[0])) / (np.max(kmeans_clusters_ica[0])-np.min(kmeans_clusters_ica[0]))
    ica_km_normed = np.expand_dims(ica_km_normed, axis=1)
    ica_new_x_train = np.append(x_train, ica_km_normed, axis=1)
    ica_km_normed = (kmeans_clusters_ica[1]-np.min(kmeans_clusters_ica[1])) / (np.max(kmeans_clusters_ica[1])-np.min(kmeans_clusters_ica[1]))
    ica_km_normed = np.expand_dims(ica_km_normed, axis=1)
    ica_new_x_test = np.append(x_test, ica_km_normed, axis=1)
    
    nn.experiment(ica_new_x_train, ica_new_x_test, y_train, y_test)

    print('\n------------')
    print('NN + kmeans_clusters_pca + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    pca_km_normed = (kmeans_clusters_pca[0]-np.min(kmeans_clusters_pca[0])) / (np.max(kmeans_clusters_pca[0])-np.min(kmeans_clusters_pca[0]))
    pca_km_normed = np.expand_dims(pca_km_normed, axis=1)
    pca_new_x_train = np.append(x_train, pca_km_normed, axis=1)
    pca_km_normed = (kmeans_clusters_pca[1]-np.min(kmeans_clusters_pca[1])) / (np.max(kmeans_clusters_pca[1])-np.min(kmeans_clusters_pca[1]))
    pca_km_normed = np.expand_dims(pca_km_normed, axis=1)
    pca_new_x_test = np.append(x_test, pca_km_normed, axis=1)
    
    nn.experiment(pca_new_x_train, pca_new_x_test, y_train, y_test)

    print('\n------------')
    print('NN + kmeans_clusters_kpca + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    kpca_km_normed = (kmeans_clusters_kpca[0]-np.min(kmeans_clusters_kpca[0])) / (np.max(kmeans_clusters_kpca[0])-np.min(kmeans_clusters_kpca[0]))
    kpca_km_normed = np.expand_dims(kpca_km_normed, axis=1)
    kpca_new_x_train = np.append(x_train, kpca_km_normed, axis=1)
    kpca_km_normed = (kmeans_clusters_kpca[1]-np.min(kmeans_clusters_kpca[1])) / (np.max(kmeans_clusters_kpca[1])-np.min(kmeans_clusters_kpca[1]))
    kpca_km_normed = np.expand_dims(kpca_km_normed, axis=1)
    kpca_new_x_test = np.append(x_test, kpca_km_normed, axis=1)
    
    nn.experiment(kpca_new_x_train, kpca_new_x_test, y_train, y_test)
    
    print('\n------------')
    print('NN + kmeans_clusters_rgp + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    rgp_km_normed = (kmeans_clusters_rgp[0]-np.min(kmeans_clusters_rgp[0])) / (np.max(kmeans_clusters_rgp[0])-np.min(kmeans_clusters_rgp[0]))
    rgp_km_normed = np.expand_dims(rgp_km_normed, axis=1)
    rgp_new_x_train = np.append(x_train, rgp_km_normed, axis=1)
    rgp_km_normed = (kmeans_clusters_rgp[1]-np.min(kmeans_clusters_rgp[1])) / (np.max(kmeans_clusters_rgp[1])-np.min(kmeans_clusters_rgp[1]))
    rgp_km_normed = np.expand_dims(rgp_km_normed, axis=1)
    rgp_new_x_test = np.append(x_test, rgp_km_normed, axis=1)
    
    nn.experiment(rgp_new_x_train, rgp_new_x_test, y_train, y_test)
    
    print('\n------------')
    print('NN + gmm_clusters_ica + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    ica_gmm_normed = (gmm_clusters_ica[0]-np.min(gmm_clusters_ica[0])) / (np.max(gmm_clusters_ica[0])-np.min(gmm_clusters_ica[0]))
    ica_gmm_normed = np.expand_dims(ica_gmm_normed, axis=1)
    ica_new_x_train = np.append(x_train, ica_gmm_normed, axis=1)
    ica_gmm_normed = (gmm_clusters_ica[1]-np.min(gmm_clusters_ica[1])) / (np.max(gmm_clusters_ica[1])-np.min(gmm_clusters_ica[1]))
    ica_gmm_normed = np.expand_dims(ica_gmm_normed, axis=1)
    ica_new_x_test = np.append(x_test, ica_gmm_normed, axis=1)
    
    nn.experiment(ica_new_x_train, ica_new_x_test, y_train, y_test)

    print('\n------------')
    print('NN + gmm_clusters_pca + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    pca_gmm_normed = (gmm_clusters_pca[0]-np.min(gmm_clusters_pca[0])) / (np.max(gmm_clusters_pca[0])-np.min(gmm_clusters_pca[0]))
    pca_gmm_normed = np.expand_dims(pca_gmm_normed, axis=1)
    pca_new_x_train = np.append(x_train, pca_gmm_normed, axis=1)
    pca_gmm_normed = (gmm_clusters_pca[1]-np.min(gmm_clusters_pca[1])) / (np.max(gmm_clusters_pca[1])-np.min(gmm_clusters_pca[1]))
    pca_gmm_normed = np.expand_dims(pca_gmm_normed, axis=1)
    pca_new_x_test = np.append(x_test, pca_gmm_normed, axis=1)
    
    nn.experiment(pca_new_x_train, pca_new_x_test, y_train, y_test)

    print('\n------------')
    print('NN + gmm_clusters_kpca + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    kpca_gmm_normed = (gmm_clusters_kpca[0]-np.min(gmm_clusters_kpca[0])) / (np.max(gmm_clusters_kpca[0])-np.min(gmm_clusters_kpca[0]))
    kpca_gmm_normed = np.expand_dims(kpca_gmm_normed, axis=1)
    kpca_new_x_train = np.append(x_train, kpca_gmm_normed, axis=1)
    kpca_gmm_normed = (gmm_clusters_kpca[1]-np.min(gmm_clusters_kpca[1])) / (np.max(gmm_clusters_kpca[1])-np.min(gmm_clusters_kpca[1]))
    kpca_gmm_normed = np.expand_dims(kpca_gmm_normed, axis=1)
    kpca_new_x_test = np.append(x_test, kpca_gmm_normed, axis=1)
    
    nn.experiment(kpca_new_x_train, kpca_new_x_test, y_train, y_test)

    print('\n------------')
    print('NN + gmm_clusters_rgp + original features')
    print('---------------')
    
    nn = NeuralNetwork(layer1_nodes=10,
                       layer2_nodes=10,learning_rate=0.05,
                       )
    
    # Add ica reduced kmeans clusters to orignal data
    rgp_gmm_normed = (gmm_clusters_rgp[0]-np.min(gmm_clusters_rgp[0])) / (np.max(gmm_clusters_rgp[0])-np.min(gmm_clusters_rgp[0]))
    rgp_gmm_normed = np.expand_dims(rgp_gmm_normed, axis=1)
    rgp_new_x_train = np.append(x_train, rgp_gmm_normed, axis=1)
    rgp_gmm_normed = (gmm_clusters_rgp[1]-np.min(gmm_clusters_rgp[1])) / (np.max(gmm_clusters_rgp[1])-np.min(gmm_clusters_rgp[1]))
    rgp_gmm_normed = np.expand_dims(rgp_gmm_normed, axis=1)
    rgp_new_x_test = np.append(x_test, rgp_gmm_normed, axis=1)
    
    nn.experiment(rgp_new_x_train, rgp_new_x_test, y_train, y_test)

       
        
        