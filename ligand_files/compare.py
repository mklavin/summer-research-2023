import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def process_random(dataframe, filename, fraction=0.75):

    train = dataframe.sample(frac=fraction)
    test = dataframe.drop(train.index)
    train.to_csv(f'{filename}_train.csv')
    test.to_csv(f'{filename}_test.csv')

    return None



def process_by_struct(dataframe, ligandsmiles, filename):
    ligand_dict = {}
    for smiles, label in zip(ligandsmiles['ligand_smiles'], ligandsmiles['index']):
        if 'O2' in smiles:
            ligand_dict[label] = 'biox'
        else:
            ligand_dict[label] = 'biim'
    dataframe['type'] = dataframe['Free Ligand'].apply(lambda x: ligand_dict[x])
    free_ligand_feat1 = dataframe.drop(columns='Free Ligand')
    biox_df = free_ligand_feat1.loc[free_ligand_feat1['type'] == 'biox']
    biim_df = free_ligand_feat1.loc[free_ligand_feat1['type'] == 'biim']

    # make 4 different ligand_files frames
    biox_train_df = biox_df.sample(frac=0.75)
    biox_test_df = biox_df.drop(biox_train_df.index)

    biim_train_df = biim_df.sample(frac=0.75)
    biim_test_df = biim_df.drop(biim_train_df.index)

    # training ligand_files
    training_set1 = [biox_train_df, biim_train_df]
    training_set = pd.concat(training_set1)
    training_set = training_set.drop(columns='type')

    # test ligand_files
    testing_set1 = [biox_test_df, biim_test_df]
    testing_set = pd.concat(testing_set1)
    testing_set = testing_set.drop(columns='type')

    # save to csv
    training_set.to_csv(f'{filename}_train.csv')
    testing_set.to_csv(f'{filename}_test.csv')

    return None




if __name__ == '__main__':
    # load ligand_files
    free_ligand_feat = pd.read_csv("free_ligand_features.csv")
    cl_aromatics_feat = pd.read_csv("L2NiArCl_features.csv")
    f2_feat = pd.read_csv("L2NiF2_features.csv")
    ligand_smiles = pd.read_csv("ligand_smiles_with_ee.csv")
    df = free_ligand_feat

    # partition
    #process_by_struct(free_ligand_feat, ligand_smiles, 'free_lig_struct')
    #process_random(free_ligand_feat, 'free_lig_rand')

    #process_by_struct(cl_aromatics_feat, ligand_smiles, 'claro_struct')
    #process_random(cl_aromatics_feat, 'claro_rand')

    #process_by_struct(f2_feat, ligand_smiles, 'f2_struct')
    #process_random(f2_feat, 'f2_rand')










