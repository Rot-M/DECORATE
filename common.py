import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


def test_measurements(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag (cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    target_names = list (set ().union (y_test.tolist(), y_pred.tolist()))
    target_names_string = ['class' + str (s) for s in target_names]
    clf_report = classification_report (y_test, y_pred, target_names=target_names_string, output_dict=True)

    # Precision
    PPV = clf_report['weighted avg']['precision']

    # true positive rate
    TPR = clf_report['weighted avg']['recall']

    # false positive rate
    FPR = FP/(FP + TN)

    # Overall accuracy
    ACC = (TP + TN)/(TP + FP + FN + TN)
    ACC_max = max(list(ACC))

    weights = cnf_matrix.sum(axis=1)/cnf_matrix.sum()
    FPR_weighted = 0
    for l in range(len(target_names)):
        FPR_weighted += weights[l]*FPR[l]

    AUC_roc = multiclass_roc_auc_score(y_test, y_pred, average="weighted")
    AUC_pr = multiclass_precision_recall_auc_score(y_test, y_pred, target_names, weights, average="weighted")
    return ACC_max, TPR, FPR_weighted, PPV, AUC_roc, AUC_pr

def y_binarize(y_test, y_pred, only_test=False):
    lb = LabelBinarizer()
    if (only_test):
        lb.fit(y_test.tolist())
    else:
        lb.fit (y_test.tolist() + y_pred.tolist())
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return y_test,y_pred

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    y_test,y_pred = y_binarize(y_test, y_pred, only_test=True)
    try :
        auc_val = roc_auc_score (y_test, y_pred, average=average)
    except ValueError :
        pass
        auc_val = 0
    return auc_val

def multiclass_precision_recall_auc_score(y_test, y_pred, target_names, weights, average="weighted"):
    if len(target_names) > 2:
        y_test, y_pred = y_binarize(y_test, y_pred)
        precision = dict()
        recall = dict()
        auc_dict = dict()
        auc_val = 0
        for l in range(len(target_names)):
            precision[l], recall[l], _ = precision_recall_curve(y_test[:, l], y_pred[:, l])
            try :
                auc_dict[l] = auc(precision[l], recall[l], reorder=True)
            except ValueError:
                pass
                auc_val = 0
            if average == "weighted":
                auc_val += weights[l]*auc_dict[l]
            else:
                auc_val += auc_dict[l]
        if average != "weighted":
            auc_val /= len(auc_dict)
    else:
        max_label = max(list(y_test)+list(y_pred))
        if max_label == 1:
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
        else:
            precision, recall, _ = precision_recall_curve(y_test, y_pred, pos_label=max_label)
        try :
            auc_val = auc(precision, recall)
        except ValueError:
            pass
            auc_val = 0

    return auc_val