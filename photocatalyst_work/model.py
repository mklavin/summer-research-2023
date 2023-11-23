import warnings
import sys
import pandas as pd
import pickle
#from photocatalyst_work.training import *
from sklearn.model_selection import cross_val_score
import glob
from padelpy import padeldescriptor
from sklearn.metrics import r2_score, mean_absolute_error
from utils import remove_emptyrows, get_variable_name, save_model
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def split_xandy(dataframe, x, y:str):
    # splits x and y
    xset = dataframe[x]
    yset = dataframe[y]
    return xset, yset

def create_trainingandtest(x, y):
    indices = np.where(y.isna())[0]
    x = np.array(x)
    x = np.delete(x, indices, axis=0)
    y = y.dropna()
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    return x_train, x_test, y_train, y_test

def evaluate_withmodels(xset, yset, models, iter=10):
    """
    :param xset: xvalues, for example, absemiswork
    :param yset: yvalues, for validation
    :param models: list of models to loop through and evaluate
    :param iter: number of times model is evaluated
    :return: prints stats on each model and returns list of stats
    """
    columns = xset.columns
    if 'Name' in columns:
        xset = xset.drop(columns='Name')
    if 'SMILES' in columns:
        xset = xset.drop(columns='SMILES')
    if 'index' in  columns:
        xset = xset.drop(columns='index')
    if 'index' in yset.columns:
        yset = yset.drop(columns='index')
    if 'Unnamed: 0' in columns:
        xset = xset.drop(columns='Unnamed: 0')

    output = np.empty([0,0])

    for model in tqdm(models, desc='model'):
        maes = []
        rmses = []
        r2s = []

        for i in tqdm(range(iter), leave=False, desc='training_iter'):
            # train test split
            x_train, x_test, y_train, y_test = create_trainingandtest(xset, yset)
            regressor = model
            #print(cross_val_score(regressor, xset, yset, cv=10, scoring='mean_absolute_error'))
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            maes.append(mae)
            r2s.append(r2)
            rmses.append(rmse)


        #print(f'MAE for {type(model).__name__}: {np.average(maes)}')
        #print(f'RMSE for {type(model).__name__}: {np.average(rmses)}')
        #print(f'R2 for {type(model).__name__}: {np.average(r2s)}')
        #, np.average(rmses), np.average(r2s)
        output = np.append(output, np.average(maes))
    return output
def remove_cluster(xset, yset, model, cluster_df, mae=True):
    """
    :param xset: input values like a fingerprint
    :param yset: output values
    :param model: model to evaluate
    :param cluster_df: dataframe containing cluster labels and indices
    :param mae: bool. If true, calculates mean absolute error for each test set; if false, returns the list of absolute
    errors
    :return: gives stats on how the model performed on the removed cluster
    """
    columns = xset.columns
    if 'Name' in columns:
        xset = xset.drop(columns='Name')
    if 'SMILES' in columns:
        xset = xset.drop(columns='SMILES')

    if mae:
        maes_allclusters = []
        maes_alltraining = []
    else:
        absolute_errors_allclusters = {}

    cluster_numbers = cluster_df['cluster']
    cluster_numbers = [*range(0, max(cluster_numbers))]

    for i in tqdm(cluster_numbers):
        xset_copy = xset.copy()
        yset_copy = yset.copy()
        clusterrows = cluster_df.loc[cluster_df['cluster'] == i]
        indices = clusterrows['index']
        indices = np.array(indices.sample(n=len(indices)-1))
        # indices holds the indices of the molecules in a cluster

        # train/test split based on cluster labels
        clusterfingers = xset_copy.iloc[indices]
        xset1 = xset_copy.loc[~xset_copy['index'].isin(indices)]
        clusteremission = yset_copy.iloc[indices]
        yset1 = yset_copy.loc[~yset_copy['index'].isin(indices)]

        # make train/test Xs, ys
        Xs = xset1.drop(['index'], axis=1).to_numpy()
        ys = yset1.drop(['index'], axis=1).to_numpy().ravel()
        Xs_test = clusterfingers.drop(['index'], axis=1).to_numpy()
        ys_test = clusteremission.drop(['index'], axis=1).to_numpy().ravel()

        # training model
        model.fit(Xs, ys)
        y_pred_cluster = model.predict(Xs_test)
        y_pred_train = model.predict(Xs)

        if mae:
            mae_cluster = mean_absolute_error(ys_test, y_pred_cluster)
            maes_allclusters.append(mae_cluster)
            mae_training = mean_absolute_error(ys, y_pred_train)
            maes_alltraining.append(mae_training)
            print(f'\ncluster{i}')
            print(f'average MAE for cluster: {mae_cluster}')
            print(f'average MAE for set without cluster: {mae_training}')
        else:
            absolute_errors_allclusters[i] = [abs(a-b) for a, b in zip(ys_test, y_pred_cluster)]

    if mae:
        return maes_allclusters, maes_alltraining
    else:
        return absolute_errors_allclusters

def evaluate_models(xset, yset, models):
    """
    :param xset: xvalues, for example, absemiswork
    :param yset: yvalues, for validation
    :param models: list of models to loop through and evaluate
    :param iter: number of times model is evaluated
    :return: prints stats on each model and returns list of stats
    """
    xcolumns = xset.columns
    ycolumns = yset.columns
    if 'Name' in xcolumns:
        xset = xset.drop(columns='Name')
    if 'SMILES' in xcolumns:
        xset = xset.drop(columns='SMILES')
    if 'index' in xcolumns:
        xset = xset.drop(columns='index')
    if 'index' in ycolumns:
        yset = yset.drop(columns='index')

    output = {}

    for model in tqdm(models, desc='model'):
        keys = np.arange(0, len(xset))
        np.random.shuffle(keys)
        xset['key'] = pd.DataFrame(keys)
        yset['key'] = pd.DataFrame(keys)
        yset = yset.sort_values(by=['key'])
        xset = xset.sort_values(by=['key'])

        # drop key column- don't want in model
        yset = yset.drop(columns='key')
        xset = xset.drop(columns='key')

        cvscore = np.average(cross_val_score(model, xset, yset, cv=5, scoring='neg_mean_absolute_error'))
        output[type(model).__name__] = cvscore
    return output

def predict(smile:list):
    """
    :param smile: smi file with just SMILES as strings
    :return: dictionary of dictionaries with first key as string, second key absp, emiss, and redox
    """
    # if this function isn't working, check directories!
    xml_files = glob.glob("redoxwork/end_to_end_xmlfiles/*.xml")
    xml_files.sort()

    FP_list = ['EState',
               'CDKextended',
               'CDK',
               'MACCS',
               'SubstructureCount',
               'Substructure']

    fp = dict(zip(FP_list, xml_files))
    smile_df = pd.DataFrame(smile, columns=['smile'])
    smile_df.to_csv('smile.smi', sep='\t', index=False, header=False)

    for i in FP_list:
        fingandxml = fp[i]
        padeldescriptor(mol_dir='smile.smi',
                        d_file=f'redoxwork/endtoend_fingerprints/{i}_fingerprint.csv',
                        descriptortypes=fingandxml,
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=-1,
                        removesalt=True,
                        log=True,
                        fingerprints=True)

    estate_fingerprint = pd.read_csv('redoxwork/endtoend_fingerprints/EState_fingerprint.csv')
    CDK_fingerprint = pd.read_csv('redoxwork/endtoend_fingerprints/CDK_fingerprint.csv')
    MACCS_fingerprint = pd.read_csv('redoxwork/endtoend_fingerprints/MACCS_fingerprint.csv')
    subcount_fingerprint = pd.read_csv('redoxwork/endtoend_fingerprints/SubstructureCount_fingerprint.csv')
    sub_fingerprint = pd.read_csv('redoxwork/endtoend_fingerprints/Substructure_fingerprint.csv')

    ox_model = pickle.load(open('finished_models/Eox_model_CDK_GBR.pkl', 'rb'))
    red_model = pickle.load(open('finished_models/Ered_model_EMACCSsub_GBR.pkl', 'rb'))
    abs_model = pickle.load(open('finished_models/Abs_model_EMACCSsub_RF.pkl', 'rb'))
    emis_model = pickle.load(open('finished_models/Emiss_model_EMACCSsub_RF.pkl', 'rb'))

    oxfingerprint = CDK_fingerprint
    MACES = [MACCS_fingerprint, estate_fingerprint, sub_fingerprint, subcount_fingerprint]
    redfingerprint = pd.concat(MACES, axis=1)
    absfingerprint = pd.concat(MACES, axis=1)
    emisfingerprint = pd.concat(MACES, axis=1)


    if 'Name' in oxfingerprint.columns:
        oxfingerprint = oxfingerprint.drop(columns='Name')
    if 'Name' in redfingerprint.columns:
        redfingerprint = redfingerprint.drop(columns='Name')
    if 'Name' in absfingerprint.columns:
        absfingerprint = absfingerprint.drop(columns='Name')
    if 'Name' in emisfingerprint.columns:
        emisfingerprint = emisfingerprint.drop(columns='Name')

    ox = ox_model.predict(oxfingerprint)
    red = red_model.predict(redfingerprint)
    abs = abs_model.predict(absfingerprint)
    emis = emis_model.predict(emisfingerprint)

    output = {}
    for i in range(len(ox)):
     output[smile[i]] = {'ox':ox[i], 'red':red[i], 'abs':abs[i], 'emis':emis[i]}

    return output

def predict_excited_state_red(absp, emis, red):
    # estimate excited state by averaging absorption and emission then add to ground state redox
    output = []
    for i in range(len(absp)):
        combined = absp[i] + emis[i]
        average = combined/2
        ev = 1239.8/average
        excitedstate = ev + red[i]
        output.append(excitedstate)
    return output


if __name__ == '__main__':
    # Excited state reduction potential from SMILES string
    smi = 'CC(C1=CC=C2C([N+](CC3=CC=C(C(C)(C)C)C=C3)=C(C=C(C(C)(C)C)C=C4)C4=C2C5=C(C)C=C(C)C=C5C)=C1)(C)C'
    properties = predict([smi]) # predict function uses previously trained models
    print(properties)

    # {'ox': 1.5510152374556185,
    #  'red': -1.5255541452663992,
    #  'abs': 462.71184848484836,
    #  'emis': 474.94991666666664,
    #  'excited state red': 1.1234458547336008,
    #  'excited state ox': -1.09798476254}
