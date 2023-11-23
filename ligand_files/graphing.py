import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

free_ligand_struct_analysis = pd.read_csv('results- LR/free_ligand_struct_analysis.csv')


#plt.plot(list(range(1,len(ligand_files['rmse'])+1)), ligand_files['rmse'])
#for i in range(100):
    #x = ligand_files.loc[i, :].values.flatten().tolist()
    #plt.plot(x)
#ax1 = plt.subplot()
#ax1.set_xticklabels(["R2", "R2", "RMSE", "MAE", "R2 Training", "RMSE Training", "MAE Training"], rotation=45)
#plt.show()

def cleandata(dataframe):
    dataframe = dataframe.drop(columns='index')
    return dataframe

def barplot_trainvtest(dataframe, title:str):
    dataframe = cleandata(dataframe)
    means = []
    stddev = []
    headers = list(dataframe.columns)
    for i in headers:
        means.append(dataframe.loc[:,i].mean())
        stddev.append(statistics.stdev(dataframe.loc[:, i]))
    means_test = means[:len(means) // 2]
    means_train = means[len(means) // 2:]
    stddev_test = stddev[:len(stddev) // 2]
    stddev_train = stddev[len(stddev) // 2:]
    barWidth = 0.25

    br1 = np.arange(len(means_test))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, means_test, color='skyblue', width=barWidth,
            edgecolor='grey', label='Testing')
    plt.errorbar(br1, means_test, yerr=stddev_test, fmt='o')
    plt.bar(br2, means_train, color='peachpuff', width=barWidth,
            edgecolor='grey', label='Training')
    plt.errorbar(br2, means_train, yerr=stddev_train, fmt='o')

    plt.xticks([r + barWidth for r in range(len(means_test))],
               ['R2', 'RMSE', 'MAE'])
    plt.legend()
    plt.title(title)
    plt.show()
    return None

def compare_rand_struct(dataframe1, dataframe2, title:str):
    # df 1 is random
    # df 2 is struct
    dataframe1 = cleandata(dataframe1)
    dataframe2 = cleandata(dataframe2)
    means1 = []
    stddev1 = []
    means2 = []
    stddev2 = []
    headers = list(dataframe1.columns)
    for i in headers:
        means1.append(dataframe1.loc[:, i].mean())
        stddev1.append(statistics.stdev(dataframe1.loc[:, i]))
        means2.append(dataframe2.loc[:, i].mean())
        stddev2.append(statistics.stdev(dataframe2.loc[:, i]))

    barWidth = 0.25
    br1 = np.arange(len(means1))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, means1, color='skyblue', width=barWidth,
            edgecolor='grey', label='Randomly Chosen Ligands')
    plt.errorbar(br1, means1, yerr=stddev1, fmt='o')
    plt.bar(br2, means2, color='peachpuff', width=barWidth,
            edgecolor='grey', label='Ligands Chosen by Structure')
    plt.errorbar(br2, means2, yerr=stddev2, fmt='o')

    plt.xticks([r + barWidth for r in range(len(means1))],
               ['R2 Test', 'RMSE Test', 'MAE Test', 'R2 Train', 'RMSE Train', 'MAE Train'])
    plt.legend()
    plt.title(title)
    plt.show()
    return None

def compare_features(dataframe1, dataframe2, dataframe3, title:str):
    dataframe1 = cleandata(dataframe1)
    dataframe2 = cleandata(dataframe2)
    dataframe3 = cleandata(dataframe3)
    means1 = []
    stddev1 = []
    means2 = []
    stddev2 = []
    means3 = []
    stddev3 = []
    headers = list(dataframe1.columns)
    for i in headers:
        means1.append(dataframe1.loc[:, i].mean())
        stddev1.append(statistics.stdev(dataframe1.loc[:, i]))
        means2.append(dataframe2.loc[:, i].mean())
        stddev2.append(statistics.stdev(dataframe2.loc[:, i]))
        means3.append(dataframe3.loc[:, i].mean())
        stddev3.append(statistics.stdev(dataframe3.loc[:, i]))
    barWidth = 0.25
    br1 = np.arange(len(means1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, means1, color='skyblue', width=barWidth,
            edgecolor='grey', label='Cl Aromatic Ligands')
    plt.errorbar(br1, means1, yerr=stddev1, fmt='o')
    plt.bar(br2, means2, color='peachpuff', width=barWidth,
            edgecolor='grey', label='Free Ligands')
    plt.errorbar(br2, means2, yerr=stddev2, fmt='o')
    plt.bar(br3, means3, color='mediumaquamarine', width=barWidth,
            edgecolor='grey', label='F2 Ligands')
    plt.errorbar(br3, means3, yerr=stddev3, fmt='o')

    plt.xticks([r + barWidth for r in range(len(means1))],
               ['R2 Test', 'RMSE Test', 'MAE Test', 'R2 Train', 'RMSE Train', 'MAE Train'])
    plt.legend()
    plt.title(title)
    plt.show()
    return None

def compare_models(dataframe1, dataframe2, dataframe3, title:str):
    dataframe1 = cleandata(dataframe1)
    dataframe2 = cleandata(dataframe2)
    dataframe3 = cleandata(dataframe3)
    means1 = []
    stddev1 = []
    means2 = []
    stddev2 = []
    means3 = []
    stddev3 = []
    headers = list(dataframe1.columns)
    for i in headers:
        means1.append(dataframe1.loc[:, i].mean())
        stddev1.append(statistics.stdev(dataframe1.loc[:, i]))
        means2.append(dataframe2.loc[:, i].mean())
        stddev2.append(statistics.stdev(dataframe2.loc[:, i]))
        means3.append(dataframe3.loc[:, i].mean())
        stddev3.append(statistics.stdev(dataframe3.loc[:, i]))
    barWidth = 0.25
    br1 = np.arange(len(means1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, means1, color='skyblue', width=barWidth,
            edgecolor='grey', label='Elastic Net Regression')
    plt.errorbar(br1, means1, yerr=stddev1, fmt='o')
    plt.bar(br2, means2, color='peachpuff', width=barWidth,
            edgecolor='grey', label='Ridge Regression')
    plt.errorbar(br2, means2, yerr=stddev2, fmt='o')
    plt.bar(br3, means3, color='mediumaquamarine', width=barWidth,
            edgecolor='grey', label='Lasso Regression')
    plt.errorbar(br3, means3, yerr=stddev3, fmt='o')

    plt.xticks([r + barWidth for r in range(len(means1))],
               ['R2 Test', 'RMSE Test', 'MAE Test', 'R2 Train', 'RMSE Train', 'MAE Train'])
    plt.legend()
    plt.title(title)
    plt.show()
    return None



if __name__ == '__main__':
    free_ligand_struct_analysis = pd.read_csv('results- LR/free_ligand_struct_analysis.csv')
    free_ligand_rand_analysis = pd.read_csv('results- LR/free_ligand_rand_analysis.csv')

    claro_struct_analysis = pd.read_csv('results- LR/claro_struct_analysis.csv')
    claro_rand_analysis = pd.read_csv('results- LR/claro_rand_analysis.csv')

    f2_struct_analysis = pd.read_csv('results- LR/f2_struct_analysis.csv')
    f2_rand_analysis = pd.read_csv('results- LR/f2_rand_analysis.csv')

    barplot_trainvtest(free_ligand_struct_analysis, 'Free Ligand Analysis- Ligands Chosen by Structure')
    #compare_rand_struct(free_ligand_rand_analysis, free_ligand_struct_analysis, 'Comparing Free Ligands Chosen by Structure versus Randomly')
    #compare_features(claro_struct_analysis, free_ligand_struct_analysis, f2_struct_analysis, 'Linear Regression Accuracy Across Different Ligands- Sorted by Structure')