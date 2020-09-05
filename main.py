import pandas as pd
import pre_processing
import os
import decorate
import gbm

PATH = 'Data/'

final_report_columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper Parameters Values: Imax, C_Size, R_Size / n_estimator, learning rate', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time milliseconds', 'Inference Time milliseconds']
df_final_report_tests_results = pd.DataFrame(columns=final_report_columns)

i = 0
data_csv_files = os.listdir(PATH)
for file_name in data_csv_files:
    print("Run algorithms on data file: " + file_name)
    df = pd.read_csv(PATH + file_name)

    print("Run pre-processing on data file: " + file_name)
    df_new = pre_processing.pre_processing(df)
    print("Finished running pre-processing on data file: " + file_name)
    df_new.to_csv(PATH+'preprocessed data.csv')

    #Prepare final_report_framework
    for row in range(i*20,i*20+20):
        df_final_report_tests_results.loc[row, 'Dataset Name'] = file_name
    for row in range (i*20, i*20+10):
        dec_row = row
        gbm_row = row + 10
        df_final_report_tests_results.loc[dec_row, 'Algorithm Name'] = 'DERCORATE'
        df_final_report_tests_results.loc[gbm_row, 'Algorithm Name'] = 'GBM'

    print("Run DECORATE algorithm on data file: " + file_name)
    df_final_report_tests_results = decorate.decorate(df_new, df_final_report_tests_results, i*20)
    print("Finished running DECORATE algorithm on data file: " + file_name)

    print("Run GBM algorithm on data file: " + file_name)
    df_final_report_tests_results = gbm.gradient_boosing(df_new, df_final_report_tests_results, i*20 + 10)
    print("Finished running GBM algorithm on data file: " + file_name)

    #Write results to final report
    df_final_report_tests_results.to_csv("tests_results.csv")

    i += 1

df_final_report_tests_results.to_csv("tests_results.csv")


