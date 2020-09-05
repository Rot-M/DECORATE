import common
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.model_selection import KFold
from datetime import datetime
from random import shuffle
import pandas as pd
import numpy as np

def gradient_boosing( df, df_tests_results, dataset_init_row_num ):
    """
    :param df: Dataframe of the data set
    :param df_tests_results: Dataframe of tests results fo final report
    :param dataset_init_row_num: Where to start the write of results for this dataset and algorithm
    :return dataset_init_row_num: Dataframe of tests results fo final report
    """

    # Definition of X and y
    columns = df.columns
    X = df.drop(columns[-1],axis=1).to_numpy()
    y = df[columns[-1]].to_numpy()


    print("Start Random Search Hyper Parameters (n_estimator, learning_rate) step")
    n_est = list(range(1, 51))
    rate = list(np.arange(10, 55)/50)
    hp_tune_trials = 50#number of iterstions needed to determine the value of hyper parameters

    # Generate Dataframe which depicts all hyper parameters combinations
    df_hp_options = pd.DataFrame(columns=['n_estimator', 'learning_rate'])
    for n in n_est:
        for r in rate:
                df_hp_options.loc[len(df_hp_options), :] = [n, r]

    # Generate Dataframe with all hyper paramteres combinations to check according to Random Search
    hp_options_list = list(range(0, len(df_hp_options)))
    shuffle(hp_options_list)
    df_hp_options_reduced = df_hp_options.ix[hp_options_list[0:hp_tune_trials]]

    df_hp_tuning = pd.DataFrame(columns=['n_estimator', 'learning_rate', 'Score']) #Dataframe which includes all accuracy scores for each tested hyper parameters combination

    iter = 1
    for index, row in df_hp_options_reduced.iterrows():
        print("hp tuning No. %d" % iter)
        Score = gradient_boosting_alg(df_tests_results, dataset_init_row_num, 3, row['n_estimator'], row['learning_rate'], X, y, tune = True)
        df_hp_tuning.loc[len(df_hp_tuning), :] = [row['n_estimator'], row['learning_rate'], Score]
        iter += 1
    print("Finished Random Search Hyper Parameters step")

    # Hyper paramters with maximum accuracy score- used for train algorithm
    n_est = list(df_hp_tuning.sort_values('Score', ascending=False).n_estimator)[0]
    l_rate= list(df_hp_tuning.sort_values('Score', ascending=False).learning_rate)[0]

    print("Start to run algorithm with 10 CV")
    df_tests_results = gradient_boosting_alg(df_tests_results, dataset_init_row_num, 10, n_estimator=n_est, learning_rate=l_rate,  X=X, y=y, tune=False)
    print("Finished Running algorithm with 10 CV")

    return df_tests_results

def gradient_boosting_alg( df_tests_results, dataset_init_row_num, k, n_estimator, learning_rate, X, y, tune ):
    """

    :param df_tests_results: Dataframe of tests results fo final report
    :param dataset_init_row_num: Where to start the write of results for this dataset and algorithm
    :param k: fold numbers for cross validation
    :param n_estimator:
    :param learning_rate:
    :param X: Dataframe of X
    :param y: labels
    :param tune: True/False according - if to implement tuning for hyper parameters
    :return: if tune=true: return final_accuracy, otherwise return df_tests_results
    """

    # CV definition
    y_pred = np.empty(len(y))
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    fold_num = 0
    for train_index, test_index in kf.split(X):
        print("fold %d" %fold_num)
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        my_ensemble = gbm(learning_rate=learning_rate, n_estimators=n_estimator)

        init_time_train = datetime.now()#The current time of fold train start
        my_ensemble.fit(X_train, y_train)
        end_time_train = datetime.now ()  # The current time of fold train end
        runtime_train_ms = (end_time_train - init_time_train).microseconds  # train runtime


        print("Start to predict test of fold No. %d" % (fold_num))
        init_time_test_inference = datetime.now()  # The current time of fold test predict start
        y_predict = my_ensemble.predict(X_test)
        end_time_test_inference = datetime.now()  # The current time of fold test predict end
        runtime_test_inference_ms = ((end_time_test_inference - init_time_test_inference).microseconds)*1000/len (y_predict)#test predict runtime
        y_pred[test_index] = y_predict
        print("Start to calculate error prediction")
        error = np.sum(y_predict != y_test)/y_predict.size
        print("Finish to calculate error prediction " + str(error))
        print("Finish to predict test of fold No. %d"%(fold_num))


        final_fold_accuracy = np.sum(y_predict == y_test)/y_predict.shape[0]
        print('Test Accuracy', final_fold_accuracy)
        if (tune == False):
            ACC, TPR, FPR, PPV, AUC_roc, AUC_pr = common.test_measurements(y_test, y_predict)

            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Cross Validation')] = fold_num
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Hyper Parameters Values: Imax, C_Size, R_Size / n_estimator, learning rate')] = [n_estimator, learning_rate]
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Accuracy')] = ACC
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('TPR')] = TPR
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('FPR')] = FPR
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Precision')] = PPV
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('AUC')] = AUC_roc
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('PR-Curve')] = AUC_pr
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Training Time milliseconds')] = runtime_train_ms
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Inference Time milliseconds')] = runtime_test_inference_ms
        fold_num += 1

    final_accuracy = np.sum(y_pred == y)/y_pred.shape[0]
    print('Test accuracy whole model: ', final_accuracy)
    if(tune):
        return final_accuracy
    else:
        return df_tests_results