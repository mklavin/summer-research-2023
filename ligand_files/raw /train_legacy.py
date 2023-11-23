
import pandas as pd

testing_set = pd.read_csv("testing.csv")
training_set = pd.read_csv("training.csv")

y_test = testing_set[['ΔΔG‡']]
x_test = testing_set.drop(columns=['ee', 'ΔΔG‡'])

y_training = training_set[['ΔΔG‡']]
x_training = training_set.drop(columns=['ee', 'ΔΔG‡'])

#now we know which ligands are which. find corresponding ligand_files
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def choose3(x_training, y_training, x_testing, y_testing):
    header = list(x_training.columns)
    header = header[1:]
    features = random.sample(header, 3)
    x_train_final = x_training[features]
    x_testing = x_testing[features]
    x_testing = x_testing.to_numpy()
    x_training = x_train_final.to_numpy()
    y_training = y_training.to_numpy()
    LR = LinearRegression()
    LR.fit(x_training, y_training)
    y_prediction = LR.predict(x_testing)
    r2val = r2_score(y_testing, y_prediction)
    return features, r2val

features, r2val = choose3(x_training, y_training, x_test, y_test)

i=0
best_models = pd.DataFrame([])
while i <100:
    features, r2val = choose3(x_training, y_training, x_test, y_test)
    features.append(r2val)
    features = pd.Series(data=features)
    best_models = pd.concat([best_models, features], axis=1)
    i=i+1

best_models = best_models.T
best_models.to_csv('bestmodels.csv')

exit()

#free_ligand_feat_withlabels.set_index('Free Ligand',inplace=True, drop=True)
#biox_ligands_array = np.array([])
#biox_ligands_array = np.zeros((len(biox_ligands), len(free_ligand_feat_withlabels.columns)))
#for i in range(len(biox_ligands)):
    #result = free_ligand_feat_withlabels.loc[biox_ligands[i]].values
    #biox_ligands_array[i,:] = result


#train ligand_files
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train, y_train)
y_prediction = LR.predict(x_test)
from sklearn.metrics import r2_score
r2val = r2_score(y_test, y_prediction)
