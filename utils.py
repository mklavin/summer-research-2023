import pandas as pd
import pickle

def add_solventfeat(fingerlist):
    labeleddata = pd.read_csv('redoxwork/Alldata_SMILES_v0.1.csv')
    bigdf = {}
    j = 0
    for i in fingerlist:
        i['Et30'] = labeleddata['Et30']
        i['SP'] = labeleddata['SP']
        i['SdP'] = labeleddata['SdP']
        i['SA'] = labeleddata['SA']
        i['SB'] = labeleddata['SB']
        j = j+1
        bigdf[f'{j}'] = i
    return bigdf

def save_model(x,y,model,output:str):
    model.fit(x,y)
    pickle.dump(model, open(output,'wb'))
    return None

def get_variable_name(variable):
    """
    :param variable: any variable
    :return: variable name as string
    """
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name

def remove_emptyrows(x, y, name):
    # missing data can generate empty rows in fingerprints
    # use this function to clean and remove
    columns = x.columns

    if 'index' in columns:
        x = x.drop(columns='index')

    if 'Name' in columns:
        if type(x['Name']) == str:
            x['Name'] = int(x['Name'])

    columns = x.columns

    # find empty rows in x
    badrows1 = x[x.isna().any(axis=1)]
    badrows1 = badrows1[columns[0]]

    badrows1 = list(badrows1)

    for i in range(len(badrows1)):
        badrows1[i] = badrows1[i] - 1

    # remove empty rows in x
    x = x.drop(badrows1)

    # remove adjacent rows in y
    y = y.drop(badrows1)

    # reset index
    x = x.reset_index().rename(columns={x.index.name:f'{name}'})
    y = y.reset_index().rename(columns={y.index.name:f'{name}'})

    return x, y

if __name__ == '__main__':
    df = pd.read_csv('absemiswork/Alldata_SMILES_v0.1.csv')
    #df2.to_csv('molecule_noindex.smi', sep='\t', index=False, header=False)

    #df = pd.read_csv('molecule_noindex.smi')
    #df.columns = ['SMILES']
    #fingerprint_csv(df, output_path='Morgan.csv')

    #photocat_data = pd.read_csv('Alldata_SMILES_v0.1.csv')
    #fingertomaccs(df['nucleophile_smiles'], output_path='maccs.csv')


    """


    # combine dfs
    CDKex = pd.read_csv('redoxwork/raw_fingerprints/CDKextended.csv')
    #Morgan = pd.read_csv('absemiswork/raw_fingerprints/Morgan.csv')
    Estate = pd.read_csv('redoxwork/raw_fingerprints/EState.csv')
    Subcount = pd.read_csv('redoxwork/raw_fingerprints/SubstructureCount.csv')
    Sub = pd.read_csv('redoxwork/raw_fingerprints/Substructure.csv')
    MACCS = pd.read_csv('redoxwork/raw_fingerprints/MACCS.csv')

    CDKES = [CDKex, Estate, Sub, Subcount]
    #MES = [Morgan, Estate, Sub, Subcount]
    MACES = [MACCS, Estate, Sub, Subcount]

    E_CDKex_sub = pd.concat(CDKES, axis=1)
    #EMorgan_sub = pd.concat(MES, axis=1)
    E_MACCS_sub = pd.concat(MACES, axis=1)

    E_CDKex_sub.to_csv('redoxwork/raw_fingerprints/E_CDKex_sub.csv')
    #EMorgan_sub.to_csv('redoxwork/raw_fingerprints/EMorgan_sub.csv')
    E_MACCS_sub.to_csv('redoxwork/raw_fingerprints/E_MACCS_sub.csv')
    """










