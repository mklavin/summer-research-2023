import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from umap import UMAP
from sklearn.cluster import DBSCAN
from PIL import Image
import random
from rdkit.Chem import Draw
from rdkit import Chem
from utils import *

def get_mean_molecin(dataframe, cls, n_per_cluster, filename):
    """
    :param dataframe: full dataframe with smiles and fingerprint
    :param cls: generated from umap
    :param n_per_cluster: number of molecules to show
    :param filename: name of file
    :return: saves file with image of random sample of molecules in a cluster
    """
    smiles = dataframe['SMILES']
    fingerprint = dataframe.drop(columns='SMILES')
    dfs = {}
    dfs['all'] = fingerprint
    mols = pd.Series(smiles.map(Chem.MolFromSmiles), index=fingerprint.index).to_frame('mol')
    cls_df = pd.DataFrame(cls)
    mols = pd.concat([cls_df, mols], axis=1).dropna()
    clustering = 'all'
    for group, data in mols.groupby(clustering):
        selected = np.random.choice(data.index, size=n_per_cluster)

        print(f"Cluster {group}, n molecules: {len(data)}")
        ms = data['mol'].loc[selected]
        x = Draw.MolsToGridImage(ms, molsPerRow=5)
        x.save(f'./absemiswork/clusterlabels/{group, filename}.png', format="PNG")
    return None

def get_random_molecin(smiles_df, clusters):
    # get random molecules in each cluster to visualize
    # saved as images

    for i in range(max(clusters['cluster'])):
        clusterrows = clusters.loc[clusters['cluster'] == i]
        indices = clusterrows['index']
        indices = np.array(indices.sample(n=len(indices)-1))
        #indices = np.array(indices)

        smiles = smiles_df.iloc[indices]
        mols = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            mols.append(mol)
        try:
            x = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(400, 400))
        except RuntimeError:
            sublist = random.sample(mols, 50)
            print(f'cluster{i} is too big!!! {len(mols)} molecules in total')
            x = Draw.MolsToGridImage(sublist, molsPerRow=6, subImgSize=(400, 400))

        x.save(f'./absemiswork/clustervisual/cluster{i}finger.png', format="PNG")
    return None

def fig2img(fig):
    # save as images
    # used to convert SMILES to images
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def make_barplot(dataframe):
    means = dataframe['MAE']
    #stddev = dataframe['stddev']

    barWidth = 0.5

    br1 = np.arange(len(means))

    plt.bar(br1, means, color='skyblue', width=barWidth,
            edgecolor='grey', label='Testing')
    #plt.errorbar(br1, means, yerr=stddev, fmt='o')

    plt.xticks([r for r in range(len(means))],
               np.arange(36))
    plt.legend()
    plt.title('MAE for Different Clusters')
    plt.show()
    return None

def make_barplot_subplots(dataframe):
    # multiple bar plots
    means = dataframe['MAE']
    clusternums = dataframe['nummolec']

    barWidth = 0.5

    br1 = np.arange(len(means))
    br2 = np.arange(len(clusternums))

    plt.subplot(2,1,1)
    plt.bar(br1, means, color='skyblue', width=barWidth,
            edgecolor='grey', label='Testing')

    plt.subplot(2,1,2)
    plt.bar(br2, clusternums, color='skyblue', width=barWidth,
            edgecolor='grey', label='Testing')

    plt.xticks([r for r in range(len(means))],
               np.arange(36))
    plt.legend()
    plt.title('MAE for Different Clusters')
    plt.show()
    return None

def make_violinplot(dataframe):
    sns.violinplot(data=dataframe)
    plt.ylabel('MAE')
    plt.title('Absolute Errors of All Molecules')
    plt.show()
    return None

def make_boxplot(data):
    plt.boxplot(data)
    plt.xlabel('Cluster #')
    plt.ylabel('MAE')
    plt.title('MAE of the First Ten Clusters')
    plt.show()
    return None

def make_boxandviolin(all_data):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # plot violin plot
    axs[0].violinplot(all_data,
                      showmeans=False,
                      showmedians=True)
    axs[0].set_title('Violin plot')

    # plot box plot
    axs[1].boxplot(all_data)
    axs[1].set_title('Box plot')

    for ax in axs:
        ax.yaxis.grid(True)
    plt.show()
    return None

def make_heatplot(df):
    # make heatplot
    modelnames = df['name']
    df = df.drop(columns='name')
    columns = df.columns
    df = np.array(df)
    df = np.around(df, decimals=2)
    fig, ax = plt.subplots()

    plt.xticks(np.arange(len(columns)), columns)
    plt.yticks(np.arange(len(modelnames)), modelnames)

    # Loop over data dimensions and create text annotations.
    for i in range(len(modelnames)):
        for j in range(len(columns)):
            text = ax.text(j, i, df[i, j],
                           ha="center", va="center", color="w")

    plt.imshow(df, vmin=0, vmax=80, cmap='inferno')
    plt.colorbar()
    plt.title('MAE of Predicted Oxidation Potential (V)',
              fontweight="bold")
    fig.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    plt.show()
    return None

def plot_and_cluster(dataframe, nneighbors):
    """
    :param dataframe: features
    :param nneighbors: UMAP parameter input
    :return: plots UMAP values and clusters based on features, also returns clustering labels
    """
    if 'SMILES' in dataframe.columns:
        dataframe = dataframe.drop(columns='SMILES')
    if 'Name' in dataframe.columns:
        dataframe = dataframe.drop(columns='Name')

    dfs = pd.DataFrame(UMAP(n_components=10, n_neighbors=nneighbors, min_dist=0.1).fit_transform(dataframe),
                                index=dataframe.index,
                                columns=["UMAP1", "UMAP2","UMAP3","UMAP4","UMAP5","UMAP6",
                                         "UMAP7","UMAP8","UMAP9","UMAP10"])

    clustering = DBSCAN(eps=.5).fit(dfs)

    # uncover to add noise
    #umap1 = np.array(dataframe['UMAP1'])
    #umap2 = np.array(dataframe['UMAP2'])
    #for i in range(len(umap1)):
        #randy = random.uniform(-1,1)
        #umap1[i] = umap1[i] + randy
    #for i in range(len(umap2)):
        #randy = random.uniform(-1,1)
        #umap2[i] = umap2[i] + randy

    # uncover to make scatterplot
    scatter_plot = sns.scatterplot(data=dataframe, x=dataframe['UMAP1'], y=dataframe['UMAP2'], alpha=0.3, palette='gist_ncar',
                    hue_order=np.random.shuffle(np.arange(len(clustering.labels_))),
                    hue=clustering.labels_).set_title(f"Neighbors= {40}, eps=5")
    sns.set(font_scale=2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('UMAP1', fontsize=16)
    plt.ylabel('UMAP2', fontsize=16)
    plt.title(label=f"Clustering on 10-D UMAP Values", fontsize=20)
    scatter_fig = scatter_plot.get_figure()
    scatter_fig.savefig('graph2.png', dpi= 1200)
    plt.show()
    return clustering.labels_ # returns cluster results

def make_predvsmodel_plot(df, x:str, y:str):
    """
    :param df: dataframe with x and y
    :param x: column name of x vals in df
    :param y: column name of y vals in df
    :return: plot of x versus y
    """
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.scatter(2.16, 1.735689825614245, c='C1', marker='o')
    ax.scatter(2.12, 1.8684639363662447, c='C2', marker='o')
    ax.scatter(2.17, 1.7807878785303353, c='C3', marker='o')
    ax.scatter(df[x], df[y])
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot([1.7, 2.3], [1.7, 2.3])
    ax.grid(visible=True, axis='both', alpha=.5)
    plt.xlabel('Experimental Excited State Reduction Potential (V)')
    plt.ylabel('Predicted Excited State Reduction Potential (V)')

    plt.show()
    return None

def make_scatterplot(df, x:str, y:str):
    # regular scatterplot with line of best fit

    plt.scatter(df[x], df[y])

    # line of best fit
    x = np.array(df[x])
    y = np.array(df[y])
    a, b = np.polyfit(x, y, 1)
    plt.plot(x, a*x+b)

    plt.show()
    return None

def make_barplot_twobars(dataframe):
    # barplot with two adjacent bars for comparison
    bar1 = dataframe['MAE'] # change these based on data to graph
    bar2 = dataframe['nummolec'] # change

    barWidth = 0.5

    ys = np.array(bar1.values)
    args = np.argsort(ys)
    bar1_sorted = ys[args]
    bar2_sorted = np.array(bar2.values)[args]  # sorted with bar1 keys

    x_pos = np.arange(0, 72, 2)

    plt.figure(figsize=(15, 4))
    plt.bar(x_pos-.25, bar1_sorted, color='royalblue', width=barWidth,
            edgecolor='grey', label='MAE', align='edge')
    plt.bar(x_pos+.25, bar2_sorted, color='darkorange', width=barWidth,
            edgecolor='grey', label='# Molec.', align='edge')

    plt.xticks(x_pos, [r for r in range(len(bar1))])
    plt.legend()
    plt.title('MAE for Different Sized Clusters')
    plt.show()
    return None

def make_barplot2(dataframe):
    # regular barplot

    bar1 = dataframe['MAE']

    barWidth = 0.5

    ys = np.array(bar1.values)
    args = np.argsort(ys)
    bar1_sorted = ys[args]

    x_pos = np.arange(0, 72, 2)

    plt.figure(figsize=(15, 4))
    plt.bar(x_pos-.25, bar1_sorted, color='royalblue', width=barWidth,
            edgecolor='grey', label='MAE', align='edge')

    plt.xticks(x_pos, [r for r in range(len(bar1))])
    plt.legend()
    plt.title('MAE for Different Sized Clusters')
    plt.show()
    return None


if __name__ == '__main__':
    emiss_MAE = pd.read_csv('absemiswork/MAE_ofmodels/emiss_MAE.csv')
    absp_MAE = pd.read_csv('absemiswork/MAE_ofmodels/absp_MAE.csv')
    eox_MAE = pd.read_csv('redoxwork/Eox_modelresults.csv')
    ered_MAE = pd.read_csv('redoxwork/Ered_modelresults.csv')
    absandemiss = pd.read_csv('absemiswork/Alldata_SMILES_v0.1.csv')
    cls = pd.read_csv('absemiswork/clusterlabels/E_MACCS_sub_cluster_UMAP10D.csv')
    leave_out_clust = pd.read_csv('absemiswork/cluster_results/E_MACCS_sub_leaveoutclust.csv')

    make_barplot_twobars(leave_out_clust)
    make_barplot2(leave_out_clust)




