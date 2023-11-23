import random
import pandas as pd
import numpy as np
import rdkit.Chem.PandasTools
from training import remove_emptyrows
from sklearn.feature_selection import VarianceThreshold
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import DataStructs
import glob
from padelpy import padeldescriptor

def generate_morganfingerprint(mols, n_bits=2048, radius=2, output_path=''):
    """
    :param mols: smiles
    :param n_bits: size of fingerprint
    :param output_path: path of output file
    :return: None, just saves a file with morgan fingerprint
    """
    mol_df = pd.DataFrame(mols, columns=['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(mol_df, smilesCol='SMILES', molCol='ROMol')
    assert mol_df.isnull().sum().sum() == 0, 'some rdkit mol files fail to generate'
    # featurize with morgan FP
    mol_df['morganFP'] = mol_df.apply(lambda x: GetMorganFingerprintAsBitVect(x['ROMol'], radius=radius, nBits=n_bits), axis=1)
    mol_df = mol_df.drop(['ROMol'], axis=1)  # faster lookup
    mol_df = mol_df.set_index('SMILES')  # use SMILES as df index
    cols = ['']*n_bits
    df = pd.DataFrame(columns=cols, index=mol_df.index)
    for index, row in mol_df.iterrows():  # not ideal, but only run it once to create full set, okay
        fp = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(row['morganFP'], fp)
        df.loc[index] = list(fp)
    assert df.isnull().sum().sum() == 0
    # save to csv
    if output_path is not None:
        df.to_csv(output_path)  # with index (name)
    return None

def generate_allfingerprints(smi_directory: str):
    """
    :param smi_directory: path to file containing SMILES
    :return: saves file with all smiles, returns None
    """
    xml_files = glob.glob("all_xmlfiles/*.xml")
    xml_files.sort()

    FP_list = ['AtomPairs2DCount',
               'AtomPairs2D',
               'EState',
               'CDKextended',
               'CDK',
               'CDKgraphonly',
               'KlekotaRothCount',
               'KlekotaRoth',
               'MACCS',
               'PubChem',
               'SubstructureCount',
               'Substructure']

    fp = dict(zip(FP_list, xml_files))

    for i in FP_list:
        # create argument with output file?
        fingerprint_output_file = ''.join([i, '.csv'])
        fingerprint_descriptortypes = fp[i]

        padeldescriptor(mol_dir=smi_directory,
                        d_file=fingerprint_output_file,
                        descriptortypes=fingerprint_descriptortypes,
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=-1,
                        removesalt=True,
                        log=True,
                        fingerprints=True)
    return None

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

def find_eps(dataframe):
    dists = []
    for i in range(10):
        x = random.randint(0, len(dataframe))
        y = random.randint(0, len(dataframe))

        row1 = dataframe.iloc[[x]]
        row2 = dataframe.iloc[[y]]

        row1 = np.array(row1)
        row2 = np.array(row2)

        dist = np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip(row1, row2)]))

        dists.append(dist)
    return dists

def save_df_to_smi(df, filename:str):
    # save df as a smi file for converting to fingerprints
    df.to_csv(f'{filename}.smi', sep='\t', index=False, header=False)
    return None

if __name__ == '__main__':
    generate_allfingerprints('absemiswork/Alldata_SMILES_v0.1.csv')








