import pandas as pd
import numpy as np

from xgboost import XGBClassifier as xgb
from xgboost import DMatrix

import common

import shap
import matplotlib.pyplot as plt

def build_meta_classifier():
    """
    This function generates class according to tests results, the class is defined according to highest mean accuracy, if the mean accuracy for DECORATE is greater than GBM
     the class=1, otherwise the class=0
    """

    print("Start to add the class according to tests results (accuracy)")
    df_meta = pd.read_csv('ClassificationAllMetaFeatures.csv')
    df_meta['Class'] = 0

    df_tests_results = pd.read_csv('tests_results.csv')
    for i in range(len(df_meta)):
        filename = df_meta.iloc[i]['dataset']
        filename = filename +'.csv'

        auc_gbm = df_tests_results['AUC'][(df_tests_results['Algorithm Name'] == 'GBM') & (df_tests_results['Dataset Name'] == filename)].mean()
        auc_decorate = df_tests_results['AUC'][(df_tests_results['Algorithm Name'] == 'DERCORATE') & (df_tests_results['Dataset Name'] == filename)].mean()
        if (auc_decorate>auc_gbm):
            dataset_class = 1#Class 1 is for DECORATE
        else:
            dataset_class = 0#Class 1 is for GBM
        df_meta.iloc[i, df_meta.columns.get_loc('Class')] =dataset_class

    df_meta.to_csv('meta_added_class.csv')
    print("Finish to add the class according to tests results (accuracy)")


def meta_classifier():
    print("Start step of features importance")
    df_meta = pd.read_csv('meta_added_class.csv')
    feature_importance_measures = ['gain', 'weight', 'cover']
    importance_results = pd.DataFrame(columns=feature_importance_measures)
    for importance_key in feature_importance_measures:
        classifier = xgb(importance_type=importance_key)
        X = df_meta.to_numpy()[:, 2:-1] #X includes all features
        y = df_meta.to_numpy()[:, -1] #y includes class (=algorithm)
        classifier.fit(X,y)
        importance_results[importance_key] = classifier.feature_importances_

    dmat = DMatrix(X)
    shap = classifier.get_booster().predict(dmat,pred_contribs = True)
    np.savetxt('shap.csv',shap)

    importance_results.to_csv('importance results.csv')
    print("Finished step of features importance")

def test_meta_classifier():
    print("Start step of classes prediction")
    df_meta = pd.read_csv('meta_added_class.csv')
    X = df_meta.to_numpy()[:, 2:-1]
    y = df_meta.to_numpy()[:, -1].astype(float)
    y_predicted = np.zeros(y.shape[0], dtype =float)
    for i in range(X.shape[0]):
        classifier = xgb()
        map = np.ones(X.shape[0], dtype = bool)
        map[i] = False

        X_all = X[map,:]
        y_all = y[map]
        classifier.fit(X_all, y_all)
        X_to_predict = np.zeros((1,X.shape[1]))
        X_to_predict[0] = X[i] #data set in row i
        y_predicted[i] = classifier.predict(X_to_predict)
    np.savetxt('y_predicted.csv', y_predicted)

    ACC, TPR, FPR, PPV, AUC_roc, AUC_pr = common.test_measurements(y, y_predicted)
    precision, recall, _ = common.precision_recall_curve(y, y_predicted)
    AUC_pr = common.auc(recall, precision)
    meta_results = pd.DataFrame(columns = ['ACC', 'TPR', 'FPR', 'PPV', 'AUC_roc', 'AUC_pr'])
    meta_results.loc[len (meta_results)] = [ACC, TPR, FPR, PPV, AUC_roc, AUC_pr]
    meta_results.to_csv('meta_results.csv')
    print("Finished step of classes prediction")


def graphs():
    print("start to generate charts for Meta Learning step")
    shap_values = np.loadtxt('shap.csv')
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.title ('Importance Features')
    plt.savefig('importance_features.png', bbox_inches='tight')
    plt.close()
    importance_results = pd.read_csv('importance results.csv').to_numpy()

    plt.title('GAIN')
    plt.bar(importance_results[:,0],importance_results[:,1])
    plt.savefig('gain.png')
    plt.close()

    plt.title('WEIGHT')
    plt.bar(importance_results[:, 0], importance_results[:, 2])
    plt.title('weight')
    plt.savefig('weight.png')
    plt.close()

    plt.bar(importance_results[:, 0], importance_results[:, 3])
    plt.title('COVER')
    plt.savefig('cover.png')
    plt.close()

    print("start to generate charts for Meta Learning steps")



print('Meta Learner beginning')
build_meta_classifier()
meta_classifier()
test_meta_classifier()
graphs()
print('Meta Learner finished')