# A Sklearn python script
#!/usr/bin/env python3
import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn import model_selection,metrics
from sklearn import svm, naive_bayes, tree, ensemble, neighbors, linear_model
from sklearn import gaussian_process, neural_network
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import LocallyLinearEmbedding

def train_models(models, K):
    for model, model_tuple in models.items():
        (clf_gen, params) = model_tuple
        rand_search = model_selection.RandomizedSearchCV(
                    clf_gen(),
                    param_distributions=params,
                    scoring='roc_auc',
                    cv=K,
                    n_iter=30)
        rand_search.fit(input_data, input_target)
        clf = clf_gen(**rand_search.best_params_)
        scores = sk.model_selection.cross_val_score(
                    clf,
                    input_data,
                    input_target,
                    cv=K,
                    n_jobs=-1,
                    scoring='roc_auc')
        avg = scores.mean()
        print("Model: %s, avg: %s, best params: %s" %
            (model, avg, rand_search.best_params_))

input_df = pd.read_csv("./train.csv")
        
# Last column is Class, so this needs to be removed from the data
raw_data = np.delete(input_df.values, 0, 1)

# Keep class col as the target       
input_target = input_df.values[:,0]

#Feature selections
input_sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
input_data = input_sel.fit_transform(raw_data)
#lle = LocallyLinearEmbedding(n_components=40, n_neighbors=10, random_state=42)
#input_data = lle.fit_transform(raw_data)

models = {

    'svm': (svm.SVC,
        {'C': sp.stats.randint(low=1, high=100),
         'gamma':[0.1,0.05,0.01,0.005,0.001],
         'kernel': ['rbf', 'linear']
         }),

    'GTB': (ensemble.GradientBoostingClassifier,
        {'max_depth': sp.stats.randint(low=5, high=12),
        'max_features': sp.stats.randint(low=5, high=50),
        'n_estimators': sp.stats.randint(low=5, high=200),
        'learning_rate': [0.1,0.075,0.05,0.025,0.01,0.005]
         }),

    'MLP': (neural_network.MLPClassifier,
        {'hidden_layer_sizes': sp.stats.randint(low=5, high=100),
        'activation': ["relu","tanh","logistic"],
        'alpha': sp.stats.expon(scale=.01),
        'learning_rate_init': [0.075,0.05,0.025,0.01,0.005],
        'solver': ["adam"],
         }),

    'forest': (ensemble.RandomForestClassifier,
        {'n_estimators': sp.stats.randint(low=5, high=100),
         'max_depth': sp.stats.randint(low=5, high=20)
         }),

}
#train_models(models, 5)


#B Training and testing final classifier example
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

pd_data = pd.read_csv("./train.csv")
X = np.delete(pd_data.values, 0, 1)
y = pd_data.values[:,0]
test_data = pd.read_csv("./test.csv")
raw_test = np.delete(test_data.values, 0, 1)
X_test = input_sel.transform(raw_test)
#X_test = lle.transform(raw_test)

#clf = neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=35,weights='distance')
#clf = ensemble.RandomForestClassifier(n_estimators=87,max_depth=14)
#clf = SVC(C=4, gamma=0.01, kernel='rbf',probability=True)
clf = ensemble.GradientBoostingClassifier(learning_rate=0.01,n_estimators=710,max_depth=8,max_features=13)
#clf = neural_network.MLPClassifier(hidden_layer_sizes=(92),learning_rate_init=0.01,solver='adam',alpha=0.001)

clf.fit(input_data, input_target)
predictions = clf.predict_proba(X_test)
output = [x[1] for x in predictions]

# Write prediction to CSV file for submission on kaggle
output_df = pd.DataFrame(output)
output_df.index += 1
output_df.to_csv("./out.csv", float_format='%.2f')
print("Evaluation started")

scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
print(scores.mean())
#Use cross validation to evaluate the model and give score
