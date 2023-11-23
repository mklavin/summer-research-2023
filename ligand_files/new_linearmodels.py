import pandas as pd
import numpy as np
import itertools
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


def splitxandy(dataframe, x, y:str):
    # splits x and y
    xset = dataframe[list(x)]
    yset = dataframe[y]
    return xset, yset

def root_mse(y_test, y_prediction):
    rmse = math.sqrt(mean_squared_error(y_test, y_prediction))
    return rmse

def modeling_data(train_set, test_set, model, feature_list, y):
    #assert len(feature_list) == 3
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
    r2.append(r2_score(y_test, y_prediction))
    r2_train.append(r2_score(y_train,  model.predict(x_train)))
    rmse.append(root_mse(y_test, y_prediction))
    rmse_train.append(root_mse(y_train,  model.predict(x_train)))
    mae.append(mean_absolute_error(y_test, y_prediction))
    mae_train.append(mean_absolute_error(y_train,  model.predict(x_train)))
    return r2, rmse, mae, r2_train, rmse_train, mae_train

def choose_n_features(features, n=3):
    feature_lists = []
    if 'Free Ligand' in features:
        features.remove('Free Ligand')
    if 'type' in features:
        features.remove('type')
    feature_lists = list(itertools.combinations(features, n))
    return feature_lists

def savetocsv(train_set, test_set, feature_lists, filename:str):
    results = np.zeros((len(feature_lists), 6))
    for ii in range(len(feature_lists)):
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        r2, rmse, mae, r2_train, rmse_train, mae_train = modeling_data(train_set=train_set, test_set=test_set,
                                                                     model=model, feature_list=feature_lists[ii], y='ee')
        results[ii, :] = [*r2, *rmse, *mae, *r2_train, *rmse_train, *mae_train]
    # uncover to view coefficients
    #print(model.coef_)
    #exit()
    results_df = pd.DataFrame(results, columns=['r2', 'rmse', 'mae', 'r2_train', 'rmse_train', 'mae_train', ])
    results_df.to_csv(filename)
    return None

def savetocsv_allfeatures(train_set, test_set, feature_lists, filename:str):
    results = np.zeros((len(feature_lists), 6))
    model = Lasso(alpha=0.1, max_iter=2000)
    r2, rmse, mae, r2_train, rmse_train, mae_train = modeling_data(train_set=train_set, test_set=test_set,
                                                                     model=model, feature_list=feature_lists[0], y='ee')
    results[:] = [*r2, *rmse, *mae, *r2_train, *rmse_train, *mae_train]
    # uncover to view coefficients
    print(model.coef_)
    exit()
    results_df = pd.DataFrame(results, columns=['r2', 'rmse', 'mae', 'r2_train', 'rmse_train', 'mae_train', ])
    results_df.to_csv(filename)
    return None

if __name__ == '__main__':
    # load relevant ligand_files
    free_ligand_struct_analysis = pd.read_csv('results- LR/free_ligand_struct_analysis.csv')
    free_ligand_rand_analysis = pd.read_csv('results- LR/free_ligand_rand_analysis.csv')

    claro_struct_analysis = pd.read_csv('results- LR/claro_struct_analysis.csv')
    claro_rand_analysis = pd.read_csv('results- LR/claro_rand_analysis.csv')

    f2_struct_analysis = pd.read_csv('results- LR/f2_struct_analysis.csv')
    f2_rand_analysis = pd.read_csv('results- LR/f2_rand_analysis.csv')

    # make list of features to choose from
    all_features = pd.read_csv('trainandtest/free_lig_struct_train.csv')
    all_features = [list(all_features.columns.values[3:])]

    # dataframe of all combinations of features
    #feature_lists = choose_n_features(list(all_features))

    # load training ligand_files and test set ligand_files

    # train_set = pd.read_csv('ligand_files/free_lig_rand_train.csv')
    # test_set = pd.read_csv('ligand_files/free_lig_rand_test.csv')

    train_set = pd.read_csv('trainandtest/free_lig_struct_train.csv')
    test_set = pd.read_csv('trainandtest/free_lig_struct_test.csv')

    # train_set = pd.read_csv('ligand_files/claro_rand_train.csv')
    # test_set = pd.read_csv('ligand_files/claro_rand_test.csv')

    # train_set = pd.read_csv('ligand_files/claro_struct_train.csv')
    # test_set = pd.read_csv('ligand_files/claro_struct_test.csv')

    # train_set = pd.read_csv('ligand_files/f2_rand_train.csv')
    # test_set = pd.read_csv('ligand_files/f2_rand_test.csv')

    # train_set = pd.read_csv('ligand_files/f2_struct_train.csv')
    # test_set = pd.read_csv('ligand_files/f2_struct_test.csv')

    # use model
    #savetocsv_allfeatures(train_set, test_set, all_features, 'results_Lasso/free_lig_struct_analysis_allfeat_lasso.csv')
    #savetocsv(train_set, test_set, feature_lists, 'results_ElasticNet/free_lig_struct_analysis_elasticnet1.csv')

    # import results when done

    # Ridge
    free_lig_rand_analysis_RLM = pd.read_csv('results- regularized LM/free_lig_rand_analysis_RLM.csv')
    free_lig_struct_analysis_RLM = pd.read_csv('results- regularized LM/free_lig_struct_analysis_RLM.csv')

    f2_rand_analysis_RLM = pd.read_csv('results- regularized LM/f2_rand_analysis_RLM.csv')
    f2_struct_analysis_RLM = pd.read_csv('results- regularized LM/f2_struct_analysis_RLM.csv')

    claro_rand_analysis_RLM = pd.read_csv('results- regularized LM/claro_rand_analysis_RLM.csv')
    claro_struct_analysis_RLM = pd.read_csv('results- regularized LM/claro_struct_analysis_RLM.csv')

    # Lasso
    free_lig_rand_analysis_lasso = pd.read_csv('results_Lasso/free_lig_rand_analysis_lasso.csv')
    free_lig_struct_analysis_lasso = pd.read_csv('results_Lasso/free_lig_struct_analysis_lasso.csv')

    f2_rand_analysis_lasso = pd.read_csv('results_Lasso/f2_rand_analysis_lasso.csv')
    f2_struct_analysis_lasso = pd.read_csv('results_Lasso/f2_struct_analysis_lasso.csv')

    claro_rand_analysis_lasso = pd.read_csv('results_Lasso/claro_rand_analysis_lasso.csv')
    claro_struct_analysis_lasso = pd.read_csv('results_Lasso/claro_struct_analysis_lasso.csv')

    # Elastic Net
    free_lig_rand_analysis_elasticnet = pd.read_csv('results_Elasticnet/free_lig_rand_analysis_elasticnet.csv')
    free_lig_struct_analysis_elasticnet = pd.read_csv('results_Elasticnet/free_lig_struct_analysis_elasticnet.csv')

    f2_rand_analysis_elasticnet = pd.read_csv('results_Elasticnet/f2_rand_analysis_elasticnet.csv')
    f2_struct_analysis_elasticnet = pd.read_csv('results_Elasticnet/f2_struct_analysis_elasticnet.csv')

    claro_rand_analysis_elasticnet = pd.read_csv('results_Elasticnet/claro_rand_analysis_elasticnet.csv')
    claro_struct_analysis_elasticnet = pd.read_csv('results_Elasticnet/claro_struct_analysis_elasticnet.csv')


    # graph results
    #print(barplot_trainvtest(claro_struct_analysis_RLM, "woo"))
    #print(compare_rand_struct(free_lig_rand_analysis_RLM, free_lig_struct_analysis_RLM, 'lol'))
    #print(compare_features(claro_struct_analysis_RLM, free_lig_struct_analysis_RLM, f2_struct_analysis_RLM,
                           #'Ridge Regression Accuracy Across Different Ligands- Sorted by Structure'))
    #print(compare_models(free_lig_struct_analysis_elasticnet, free_lig_struct_analysis_RLM, free_lig_struct_analysis_lasso,
                         #'Comparing Across Models- Free Ligands Chosen by Structure'))