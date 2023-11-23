import pandas as pd


free_ligand_feat = pd.read_csv("free_ligand_features.csv")
cl_aromatics_feat = pd.read_csv("L2NiArCl_features.csv")
f2_feat = pd.read_csv("L2NiF2_features.csv")

#practice
#mean, std dev, etc.
free_ligand_feat.describe()

#select specific column
gibbs = free_ligand_feat['ΔΔG‡']

import matplotlib.pyplot as plt
#free_ligand_feat.plot()
free_ligand_feat[["ee", "ΔΔG‡"]].median()

free_ligand_feat.head()

free_ligand_feat.pivot_table(values="HOMO", index="ΔΔG‡", columns="ee", aggfunc="mean", margins=True)

import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(free_ligand_feat, .33)

#visualize ligand_files
print(list(free_ligand_feat.columns))
free_ligand_feat.plot(x="ΔΔG‡", y="ee")
#plt.show()

free_ligand_stats = free_ligand_feat.describe()

free_ligand_feat.hist(bins=50, figsize=(12,8))

#create test set
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(free_ligand_feat, .33)

np.random.seed(42)

#normalize columns
headers = list(free_ligand_feat.columns)
headers = headers[1:]
for i in headers:
    maxval = free_ligand_feat[i].max()
    free_ligand_feat[i] = free_ligand_feat[i]/maxval

import copy
free_ligand_feat_withlabels = copy.deepcopy(free_ligand_feat)
ligand_names = free_ligand_feat.pop('Free Ligand')

#correlations
corr_matrix = free_ligand_feat.corr()
corr_matrix["ee"].sort_values(ascending=False)

#testing and training set
from sklearn.model_selection import train_test_split
y = free_ligand_feat['ee'].to_numpy()
data_no_ee = free_ligand_feat.drop(columns=['ee'])
x = data_no_ee.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25)

#make non-random training set based on ligand type
ligand_smiles = pd.read_csv("ligand_smiles_with_ee.csv")
ligand_dict = {}
biox_ligands = []
biim_ligands =[]
for smiles, label in zip(ligand_smiles['ligand_smiles'], ligand_smiles['index']):
    if 'O2' in smiles:
        ligand_dict[label] = 'biox'
        biox_ligands.append(label)
    else:
        ligand_dict[label] = 'biim'
        biim_ligands.append(label)

free_ligand_feat_withlabels['type'] = free_ligand_feat_withlabels['Free Ligand'].apply(lambda x: ligand_dict[x])
free_ligand_feat1 = free_ligand_feat_withlabels.drop(columns='Free Ligand')
biox_df = free_ligand_feat1.loc[free_ligand_feat1['type'] == 'biox']
biim_df = free_ligand_feat1.loc[free_ligand_feat1['type'] == 'biim']

#make 4 different ligand_files frames
biox_train_df = biox_df.sample(frac=0.8)
biox_test_df = biox_df.drop(biox_train_df.index)

biim_train_df = biim_df.sample(frac=0.8)
biim_test_df = biim_df.drop(biim_train_df.index)

#training ligand_files
training_set1 = [biox_train_df, biim_train_df]
training_set = pd.concat(training_set1)
training_set = training_set.drop(columns='type')
training_set.to_csv('training.csv')

#test ligand_files
testing_set1 = [biox_test_df, biim_test_df]
testing_set = pd.concat(testing_set1)
testing_set = testing_set.drop(columns='type')
testing_set.to_csv('testing.csv')

# common codes


def process_by_struct(dataframe):
    # code

    return None


def process_random(dataframe):


    return None

def normalize(dataset):
    headers = list(dataset.columns)
    headers = headers[1:]
    for i in headers:
        maxval = free_ligand_feat[i].max()
        free_ligand_feat[i] = free_ligand_feat[i] / maxval


if __name__ == '__main__':
    # load ligand_files
    df = ...

    # partition
    process_by_struct(dataframe=df)
    #process_random(dataframe=df)

    # normalize datasets and save