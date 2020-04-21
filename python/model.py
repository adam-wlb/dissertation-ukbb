"""model.py - Machine learning model for analysing the Biobank dataset"""
__author__ = "Adam Barron - 160212899"

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, auc, plot_roc_curve
from sklearn.datasets import load_breast_cancer

# --- Load breast cancer dataset --- #
# X, y = load_breast_cancer(return_X_y=True)
# feature_names = np.array(load_breast_cancer().feature_names)

# --- Load cleaned biobank dataset --- #
biobank = pd.read_csv('resources/biobank.csv')
# Separate features and class labels
feature_names = np.array((biobank.iloc[:, :-1]).columns)
X = biobank.iloc[:, :-1].values
y = biobank.iloc[:, -1].values

# --- Create classifier | comment out as appropriate --- #
classifier = LogisticRegression(max_iter=3000)
# classifier = LinearSVC(max_iter=30000, dual=False)
# classifier = RandomForestClassifier(max_features='sqrt')
# classifier = KNeighborsClassifier()
# --- Set up RFE --- #
rfe = RFE(estimator=classifier, step=0.1)
# KNN cannot perform RFE, used instead
# rfe = RFE(estimator=LinearSVC(max_iter=30000, dual=False), step=0.1)
# --- Create imputer --- #
imputer = SimpleImputer(missing_values=np.nan, strategy='median', verbose=1)
# --- Create scaler --- #
scaler = preprocessing.RobustScaler()
# --- Construct pipeline --- #
pipeline = Pipeline([("scaler", scaler), ("imputer", imputer), ("rfe", rfe), ("classifier", classifier)])

# --- Parameter grid for pipeline to search during grid-search cv | comment out as appropriate --- #
# LogisticRegression
param_grid = [{'rfe__n_features_to_select': [25, 50, 75, 100, 125],
               'classifier__solver': ['sag'],
               'classifier__penalty': ['l2'],
               'classifier__C': [0.1, 1, 10, 100],
               'classifier__tol': [1e-3, 1e-2]},
              {'rfe__n_features_to_select': [25, 50, 75, 100, 125],
               'classifier__solver': ['saga'],
               'classifier__penalty': ['elasticnet'],
               'classifier__l1_ratio': [0.5],
               'classifier__C': [0.1, 1, 10, 100],
               'classifier__tol': [1e-3, 1e-2]}]
# LinearSVC
# param_grid = [{'rfe__n_features_to_select': [25, 50, 75, 100, 125],
#                'classifier__C': [0.01, 0.1, 1, 10, 100],
#                'classifier__tol': [1e-3, 1e-2]}]
# RandomForestClassifier
# param_grid = [{'rfe__n_features_to_select': [25, 50, 75, 100, 125],
#                'classifier__max_depth': [2, 4, 6],
#                'classifier__min_samples_leaf': [2, 4, 8],
#                'classifier__n_estimators': [100, 200, 300]}]
# KNeighborsClassifier
# param_grid = [{'rfe__n_features_to_select': [10, 20, 30, 40, 50],
#                   'classifier__n_neighbors': [3, 5, 7, 9],
#                   'classifier__weights': ['uniform', 'distance']}]

# Performance metrics
metrics = ['f1_weighted', 'roc_auc']

# Data structures for ROC graph
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# Model scores
scores_f1 = []
scores_roc = []

# Chosen features
chosen_features = []

# Split data using k-fold
k_fold = KFold(n_splits=5, shuffle=True)
iters = 1
start = time.perf_counter()
for train_idx, test_idx in k_fold.split(X):
    print("Tuning parameters against fold %s of 5" % iters)
    # Create training and test sets for each fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Tune hyper-parameters using grid-search cross-validation

    for metric in metrics:
        print("Tuning parameters against %s" % metric)
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=4, scoring='%s' % metric)
        grid_search.fit(X_train, y_train)

        # Best estimator score, parameters and features chosen
        best_estimator = grid_search.best_estimator_
        supports = list(best_estimator.named_steps['rfe'].support_)
        features = feature_names[supports]
        chosen_features.extend(features)
        print("Chosen features ({0}) for fold {1}: {2}".format(metric, iters, features))
        print("Best parameters ({0}) for fold {1}: {2}".format(metric, iters, grid_search.best_params_))
        print("Score ({0}) for fold {1} training set: {2}".format(metric, iters, grid_search.best_score_))

        # Predict with optimised model
        prediction = best_estimator.predict(X_test)
        # Record score
        if metric in ['f1_weighted']:
            score = f1_score(y_test, prediction, average='weighted')
            print("Score ({0})  for fold {1} test set: {2}".format(metric, iters, score))
            scores_f1.append(score)
        else:
            score = roc_auc_score(y_test, prediction)
            print("Score ({0}) for fold {1} test set: {2}".format(metric, iters, score))
            scores_roc.append(score)

    # Record ROC Curve for fold
    viz = plot_roc_curve(grid_search, X_test, y_test, name='Fold {0} ROC'.format(iters), alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    iters += 1
# Record time taken
stop = time.perf_counter()

# --- Plot ROC curve for classifier --- #
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
        alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='{0} ROC Curve'.format(classifier.__class__.__name__))
ax.legend(loc="lower right")
# Save graph
plt.savefig('results/{0}_ROC_Curve'.format(classifier.__class__.__name__), bbox_inches='tight', transparent=True)

# --- Rank features based on how many times they were selected --- #
chosen_features = np.array(chosen_features)
unique_features, count_features = np.unique(chosen_features, return_counts=True)
feature_ranking = pd.DataFrame(list(zip(unique_features, count_features)),
                               columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=False)
# Save ranking
feature_ranking.to_csv('results/{0}_Feature_Ranking.csv'.format(classifier.__class__.__name__), index=False)

# Show results
plt.show()

print('\n--------------------Results--------------------')
print("{0}:".format(classifier.__class__.__name__))
print("Feature ranking:\n{0}".format(feature_ranking))
print("F1 score for each test fold: {0}".format(scores_f1))
print("Average F1 score across all test folds: {0}".format(np.average(scores_f1)))
print("ROC_AUC score for each test fold: {0}".format(scores_roc))
print("Average ROC_AUC score across all test folds: {0}".format(np.average(scores_roc)))
print("Total time taken to train classifier: {0}".format(stop - start))
print('-----------------------------------------------\n')
