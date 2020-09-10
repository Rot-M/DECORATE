# DECORATE

1. This code is an implementation of DECORATE algorithm - [decorate.py](https://github.com/Rot-M/DECORATE/blob/master/decorate.py)
2. Comparison between GBM algorithm and DECORATE algorithm was performed -  implementation of GBM exists in [gbm.py](https://github.com/Rot-M/DECORATE/blob/master/gbm.py)
3. Mann Whitney U Test was perofmed between DECORATE and GBM - [statistic_test.py](https://github.com/Rot-M/DECORATE/blob/master/statistic_test.py)
4. Performed prediction for which algorithm is better according to meta learning features - [meta_learner.py](https://github.com/Rot-M/DECORATE/blob/master/meta_learner.py)

The algorithms are run by running [main.py](https://github.com/Rot-M/DECORATE/blob/master/main.py) 

Statistic test and Meta Learning process are running from their files sperately.


**[common.py](https://github.com/Rot-M/DECORATE/blob/master/common.py) invludes measurements calculations and is needed for all files beside the statistic test.

All algorithm running results exists in folder: [Reports and Results](https://github.com/Rot-M/DECORATE/tree/master/Reports%20and%20Results)

Final reports ([results](https://github.com/Rot-M/DECORATE/blob/master/Reports%20and%20Results/tests_results.csv) and [document](https://github.com/Rot-M/DECORATE/blob/master/Reports%20and%20Results/DECORATE_Report.pdf)) exist also in folder: [Reports and Results](https://github.com/Rot-M/DECORATE/tree/master/Reports%20and%20Results)


DECORATE hyper parmaters are defined in [decorate.py](https://github.com/Rot-M/DECORATE/blob/master/decorate.py):
Imax, C_Size, R_Size

GBM hyper parmaters are defined in [gbm.py](https://github.com/Rot-M/DECORATE/blob/master/gbm.py):
rate (for learning rate), n_est
