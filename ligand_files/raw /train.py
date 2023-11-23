# imports
import itertools


def splitxandy(dataframe, x, y:str):
    # splits x and y
    xset = dataframe[list(x)]
    yset = dataframe[y]
    return xset, yset


def rsquared(y_test, y_prediction):
    r2val = r2_score(y_test, y_prediction)
    return r2val


def mean_ae(y_test, y_prediction):
    mae = mean_absolute_error(y_test, y_prediction)
    return mae


def root_mse(y_test, y_prediction):
    rmse = math.sqrt(mean_squared_error(y_test, y_prediction))
    return rmse


def train_model(train_set, test_set, model, feature_list, y):
    """
    select features from training ligand_files using feature_list
    train linear regression model
    test model with test_set
    return R^2, RMSE, MAE

    :param train_set: training set ligand_files, pandas.DataFrame
    :param test_set: test set ligand_files, pandas.DataFrame

    :param feature_list: list of three features, list
    :return:
    """
    assert len(feature_list) == 3
    r2 = []
    rmse = []
    mae = []
    r2_train = []
    rmse_train = []
    mae_train = []
    x_train, y_train = splitxandy(train_set, feature_list, y)
    x_test, y_test = splitxandy(test_set, feature_list, y)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)
    r2.append(rsquared(y_test, y_prediction))
    r2_train.append(rsquared(y_train, model.predict(x_train)))
    rmse.append(root_mse(y_test, y_prediction))
    rmse_train.append(root_mse(y_train, model.predict(x_train)))
    mae.append(mean_ae(y_test, y_prediction))
    mae_train.append(mean_ae(y_train, LR.predict(x_train)))
    return r2, rmse, mae, r2_train, rmse_train, mae_train


def choose_n_features(features, n=3):
    """
    Iteratively choose three features randomly from a list of features (all_features)
    Generate a list of list of three featrues
    e.g., [[feat1, feat2, feat3], [feat2, feat4, feat9], ...]

    :param all_features
    :return:
    """
    feature_lists = []
    if 'Free Ligand' in features:
        features.remove('Free Ligand')
    if 'type' in features:
        features.remove('type')
    # for i in range(300):
    #     features1 = random.sample(features, 3)
    #     feature_lists.append(features1)
    feature_lists = list(itertools.combinations(features, n))
    return feature_lists


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import random
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import math

    # get a list of all the features
    all_features = pd.read_csv('../trainandtest/free_lig_struct_train.csv')
    all_features = all_features.columns.values[3:]

    feature_lists = choose_n_features(list(all_features))

    # save the generated features as df
    features_df = pd.DataFrame(feature_lists, columns=['feature1', 'feature2', 'feature3'])
    features_df.to_csv('feature_sets.csv')

    # load training ligand_files and test set ligand_files
    train_set = pd.read_csv('../trainandtest/free_lig_struct_train.csv')
    test_set = pd.read_csv('../trainandtest/free_lig_struct_test.csv')

    #train_set = pd.read_csv('ligand_files/free_lig_rand_train.csv')
    #test_set = pd.read_csv('ligand_files/free_lig_rand_test.csv')

    # train_set = pd.read_csv('ligand_files/free_lig_struct_train.csv')
    #test_set = pd.read_csv('ligand_files/free_lig_struct_test.csv')

    #train_set = pd.read_csv('ligand_files/claro_rand_train.csv')
    #test_set = pd.read_csv('ligand_files/claro_rand_test.csv')

    #train_set = pd.read_csv('ligand_files/claro_struct_train.csv')
    #test_set = pd.read_csv('ligand_files/claro_struct_test.csv')

    #train_set = pd.read_csv('ligand_files/f2_rand_train.csv')
    #test_set = pd.read_csv('ligand_files/f2_rand_test.csv')

    #train_set = pd.read_csv('ligand_files/f2_struct_train.csv')
    #test_set = pd.read_csv('ligand_files/f2_struct_test.csv')

    results = np.zeros((len(feature_lists), 6))
    for ii in range(len(feature_lists)):
        r2, rmse, mae, r2_train, rmse_train, mae_train = train_model(train_set=train_set, test_set=test_set, feature_list=feature_lists[ii], y='ee')
        results[ii, :] = [*r2, *rmse, *mae, *r2_train, *rmse_train, *mae_train]

    results_df = pd.DataFrame(results, columns=['r2', 'rmse', 'mae', 'r2_train', 'rmse_train', 'mae_train', ])
    results_df.to_csv('free_lig_struct_analysis.csv')
