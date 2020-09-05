from sklearn.tree import DecisionTreeClassifier
import common
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from datetime import datetime
from random import shuffle


def decorate(df, df_tests_results, dataset_init_row_num):
    """
    :param df: Dataframe of the data set
    :param df_tests_results: Dataframe of tests results fo final report
    :param dataset_init_row_num: Where to start the write of results for this dataset and algorithm
    :return dataset_init_row_num: Dataframe of tests results fo final report
    """

    #Definition of X and y
    columns = df.columns
    X = df.drop(columns[-1],axis=1).to_numpy()
    y = df[columns[-1]].to_numpy()


    print("Start Random Search Hyper Parameters (C_Size, Imax, R_Size) step")
    C_Size = list(range(20, 51)) #Amount of learners in the ensemble
    Imax = list(range(20, 51)) #Maximum amount of iterations needed for building an ensemble
    R_Size = list(np.arange (0.1, 1, 0.1)) #Factor to determine number of artificial examples to generate
    hp_tune_trials = 50 #number of iterstions needed to determine the value of hyper parameters

    # Generate Dataframe which depicts all hyper parameters combinations
    df_hp_options = pd.DataFrame(columns=['C_Size', 'Imax', 'R_Size'])
    for c in C_Size:
        for r in R_Size:
            for i in Imax:
                df_hp_options.loc[len(df_hp_options), :] = [c, i, r]

    # Generate Dataframe with all hyper paramteres combinations to check according to Random Search
    hp_options_list = list(range(0, len(df_hp_options)))
    shuffle(hp_options_list)
    df_hp_options_reduced = df_hp_options.ix[hp_options_list[0:hp_tune_trials]]

    df_hp_tuning = pd.DataFrame(columns=['C_Size', 'Imax', 'R_Size', 'Score']) #Dataframe which includes all accuracy scores for each tested hyper parameters combination

    iter = 1
    for index, row in df_hp_options_reduced.iterrows():
        print("hp tuning No. %d" % iter)
        Score = decorate_alg(df_tests_results, dataset_init_row_num, 3, row['C_Size'], row['Imax'], row['R_Size'], X, y, tune = True)
        df_hp_tuning.loc[len(df_hp_tuning), :] = [row['C_Size'], row['Imax'], row['R_Size'], Score]
        iter += 1
    print("Finished Random Search Hyper Parameters step")

    #Hyper paramters with maximum accuracy score- used for train algorithm
    C_Size = list(df_hp_tuning.sort_values('Score', ascending=False).C_Size)[0]
    Imax = list(df_hp_tuning.sort_values('Score', ascending=False).Imax)[0]
    R_Size = list(df_hp_tuning.sort_values('Score', ascending=False).R_Size)[0]


    print("Start to run algorithm with 10 CV")
    df_tests_results = decorate_alg (df_tests_results, dataset_init_row_num, 10, Imax, C_Size, R_Size, X, y, tune=False)
    print("Finished Running algorithm with 10 CV")

    return df_tests_results


def decorate_alg(df_tests_results, dataset_init_row_num, k, Imax, C_Size, R_Size, X, y, tune):
    """

    :param df_tests_results: Dataframe of tests results fo final report
    :param dataset_init_row_num: Where to start the write of results for this dataset and algorithm
    :param k: fold numbers for cross validation
    :param Imax: Maximum amount of iterations needed for building an ensemble
    :param C_Size: Amount of learners in the ensemble
    :param R_Size: Factor to determine number of artificial examples to generate
    :param X: Dataframe of X
    :param y: labels
    :param tune: True/False according - if to implement tuning for hyper parameters
    :return: if tune=true: return final_accuracy, otherwise return df_tests_results
    """

    #CV definition
    y_pred = np.empty(len(y))
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    fold_num = 0
    for train_index, test_index in kf.split(X):
        print("fold %d" % fold_num)
        print("TRAIN  indexes:", train_index, "TEST indexes:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        learner_iter = 1
        trials = 1

        my_ensemble = ensemble()#create empty ensemble

        learner = DecisionTreeClassifier()

        init_time_train = datetime.now()#The current time of fold train start

        learner.fit(X_train, y_train)

        my_ensemble.add_learner(learner)

        print("Start to calculate first error prediction")
        y_predict = my_ensemble.predict_ensemble(X_test)
        y_pred[test_index] = y_predict
        error = np.sum(y_predict != y_test)/y_predict.size
        print("Finish to calculate first error prediction " + str(error))

        while (trials < Imax) & (learner_iter < C_Size):
            X_examples = generate_artificial_training_set(X_train, R_Size)#returns dataframe with empty labels

            if math.isnan(X_examples.max()):#write  why we put this condition, it doesn't  on the origin algorithm
                trials += 1
                continue

            y_examples = classify_examples(X_examples, my_ensemble)#Label the artificial examples
            y_examples = y_examples.astype(int)

            #Add artificial example to train fold data set
            X_train_extended = np.concatenate([X_train, X_examples])
            y_train_extended = np.concatenate([y_train, y_examples])

            #Train new learner and add it to ensemble
            learner = DecisionTreeClassifier()
            learner.fit(X_train_extended, y_train_extended)
            my_ensemble.add_learner(learner)
            print(my_ensemble.classes)

            print("Start to calculate Trial No. %d error prediction" % (trials))
            y_test_predict = my_ensemble.predict_ensemble(X_test)
            new_error = np.sum(y_test_predict != y_test) / y_test_predict.size
            print("Finish to calculate Trial No. %d error prediction" % (trials))
            print("Error for Trial No. " + str(trials) + " error equals to: " + str(new_error))

            if new_error < error:
                learner_iter += 1
                error = new_error
            else:
                my_ensemble.drop_learner()

            trials += 1

        end_time_train = datetime.now()#The current time of fold train end
        runtime_train_ms = (end_time_train - init_time_train).microseconds #train runtime

        print("Start to predict test of fold No. %d" % (fold_num))
        init_time_test_inference = datetime.now()#The current time of fold test predict start
        y_test_predict = my_ensemble.predict_ensemble(X_test)
        end_time_test_inference = datetime.now()#The current time of fold test predict end
        runtime_test_inference_ms = ((end_time_test_inference - init_time_test_inference).microseconds)*1000/len (y_test_predict)#test predict runtime
        print("Finish to predict test of fold No. %d" % (fold_num))

        final_fold_accuracy = np.sum(y_test_predict == y_test)/y_test_predict.shape[0]
        print('Test Accuracy', final_fold_accuracy)

        if (tune == False):
            ACC, TPR, FPR, PPV, AUC_roc, AUC_pr = common.test_measurements(y_test, y_test_predict)

            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Cross Validation')] = fold_num
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Hyper Parameters Values: Imax, C_Size, R_Size / n_estimator, learning rate')] = [Imax, C_Size, R_Size]
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Accuracy')] = ACC
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('TPR')] = TPR
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('FPR')] = FPR
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Precision')] = PPV
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('AUC')] = AUC_roc
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('PR-Curve')] = AUC_pr
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Training Time milliseconds')] = runtime_train_ms
            df_tests_results.iat[dataset_init_row_num + fold_num, df_tests_results.columns.get_loc('Inference Time milliseconds')] = runtime_test_inference_ms
        fold_num += 1

    final_accuracy = np.sum (y_pred == y)/y_pred.shape[0]
    print('Test accuracy whole model: ', final_accuracy)

    if(tune):
        return final_accuracy
    else:
        return df_tests_results


# adds new artificial instances, with empty classes
def generate_artificial_training_set(X_train, R_size):
    """
    :param X_train: X of train fold
    :param R_size: Factor to determine number of artificial examples to generate
    :return: examples
    Generate examples with the following behavior:
        Categorical Feature: generate binary column from the same distribution as in the origin column
        Numeric Feature: generate values according to Gaussian Distribution with STD and MEAN of origin train numeric column
    """
    examples = np.zeros(shape=(int(R_size*X_train.shape[0]), X_train.shape[1]))
    for column in range(X_train.shape[1]):
        column_values_count = pd.value_counts(X_train[:, column])
        if column_values_count.isnull().any(): #Replace column with 0 when all values are only null
            examples[:, column] = 0
        if column_values_count.size <= 2:# Categorical columns are converted to hot vectors, therefore they appear as binary columns, Generate numbers from binomial distribution
            if column_values_count.size == 1:
                s = list(column_values_count.index)[0]
            else:
                p = column_values_count[list(column_values_count.index)[0]]/(column_values_count[list(column_values_count.index)[0]]+column_values_count[list(column_values_count.index)[1]])
                s1 = np.random.binomial(1, p, size=int(R_size*X_train.shape[0]))
                s2 = 1-s1
                s1 = s1.astype (float)
                s2 = s2.astype (float)
                s1[s1 >= 1] = list (column_values_count.index)[0]
                s2[s2>=1]=list(column_values_count.index)[1]
                s = s1+s2
            examples[:, column] = s

        else:# Numerical columns: generate values according to Gaussian Distribution with STD and MEAN of origin train numeric column
            std = X_train[:, column].std() #standard deviation of origin train numeric column
            mean = X_train[:, column].mean() #mean of origin train numeric column
            examples[:, column] = np.random.normal(loc=mean, scale=std, size=(int(R_size*X_train.shape[0])))
    return examples


def classify_examples(X_examples, my_ensemble):
    """
    :param X_examples
    :param my_ensemble: Ensemble of learners
    :return: examples_label
    Add the label of examples, while using reverse histogram
    """

    y_probabilities = my_ensemble.predict_proba_ensemble(X_examples)

    # Replace 0/nan  prediction with small number
    y_probabilities[y_probabilities == 0] = 0.00001
    y_probabilities[np.isnan(y_probabilities)] = 0.00001

    new_y = np.empty((y_probabilities.shape[0]))
    for j in range(y_probabilities.shape[0]):
        norm = np.linalg.norm(y_probabilities[j,:])
        y_probabilities[j] = y_probabilities[j]/norm
        opposite_probabilities = (1/y_probabilities[j])/np.sum(1/y_probabilities[j])
        opposite_probabilities[np.isnan (opposite_probabilities)] = 0.00001

        label = opposite_probabilities.tolist().index(max(opposite_probabilities.tolist())) #The label is set according to max probability
        new_y[j] = label

    examples_label = new_y
    return examples_label




class ensemble():
    """
    holds the ensemble of learners
    """
    def __init__(self):
        """
        classes: list of classes in the ensemble
        C: list of learners in ensemble
        """
        self.classes = []
        self.C = []

    def add_learner(self, learner: DecisionTreeClassifier):
        """
        :param learner: learner to add to the ensemble
        The function add leaner to ensemble by adding the classes to classes list and the leaner to C list
        """
        self.C.append(learner)
        new_classes = [i for i in learner.classes_.tolist() if i not in self.classes]
        self.classes.extend(new_classes)

    def predict_ensemble(self, X):
        """
        :param X:
        :return:
        """
        if math.isnan(X.max()):
            print('nan')
        ensemble_probabilities = np.empty((len(self.C), len(X), len(self.classes)))
        i = 0
        for learner in self.C:
            predict = learner.predict_proba(X)
            if len(self.classes) > len(learner.classes_):
                predict_shape_adapted = np.zeros(ensemble_probabilities.shape)
                j = 0
                for label in list(set(self.classes)):
                    if label in learner.classes_.tolist():
                        learner_label_index = learner.classes_.tolist().index(label)
                        predict_shape_adapted[i][:, j] = predict[:, learner_label_index]
                    else:
                        predict_shape_adapted[i][:, j] = np.zeros(len(X))
                    j += 1
            else:
                ensemble_probabilities[i] = predict
            i += 1
        y = np.mean (ensemble_probabilities, axis=0)
        y_label = np.empty(len(y))
        for row in range(len(y)):
            y_label[row] = np.argmax(y[row])
        return y_label

    def drop_learner(self):
        """"
        The function removes last learner from ensemble
        """
        if len(self.C) > 1:
            del self.C[-1]

    # returns 2d array. len(x) rows of probabilities of each class,
    def predict_proba_ensemble(self, X):
        """
        :param X:
        :return: mean prediction of all learners predictions
        """
        ensemble_results = np.empty((len(self.C), len(X), len(self.classes)))
        if math.isnan(X.max()):
            print('nan')
        i = 0
        for learner in self.C:
            predict = learner.predict_proba(X)
            if len(self.classes) > len(learner.classes_):
                predict_shape_adapted = np.zeros(ensemble_results.shape)
                j = 0
                for label in list(set(self.classes)):
                    if label in learner.classes_.tolist():
                        learner_label_index = learner.classes_.tolist().index(label)
                        predict_shape_adapted[i][:, j] = predict[:, learner_label_index]
                    else:
                        predict_shape_adapted[i][:, j] = np.zeros(len(X))
                    j += 1
            else:
                ensemble_results[i] = predict
            i += 1
        y = np.mean(ensemble_results, axis=0)#mean of all learners predictions for each class
        return y