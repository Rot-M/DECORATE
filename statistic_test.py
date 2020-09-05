# Mann-Whitney U test
from scipy.stats import mannwhitneyu
import pandas as pd

def mannwhitneyu_test(df_tests_results):
    print("Start to perform statistic Test")
    decorate_alg_accuracy = df_tests_results.loc[df_tests_results['Algorithm Name'] == 'DERCORATE', 'Accuracy' ]
    lgb_alg_accuracy = df_tests_results.loc[df_tests_results['Algorithm Name'] == 'GBM', 'Accuracy' ]

    res = []
    stat, p = mannwhitneyu(decorate_alg_accuracy, lgb_alg_accuracy)
    res.append('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
        res.append('Same distribution (fail to reject H0) with alpha=%.3f' % alpha)
    else:
        res.append('Different distribution (reject H0) with alpha=%.3f' % alpha)

    File_object = open("Statistic_test.txt","a")
    for row in res:
        File_object.write(row)
        File_object.write('\n')
    File_object.close()

    print ("Finish to perform statistic Test")
    return True

df_tests_results = pd.read_csv('tests_results.csv')
mannwhitneyu_test(df_tests_results)