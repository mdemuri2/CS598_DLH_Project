# CS598_DLH_Project
#### Maria DeMuri | NetID: mdemuri2 | mdemuri2@illinois.edu   
#### Salman Yousaf | NetID: syousaf2 | syousaf2@illinois.edu
## Deep Learning Health Care Project: Statistical supervised meta-ensemble algorithm for medical record linkage

This repository contains the full code for the CS 598 Deep Learning for Healthcare project. The main goal of the project was to reproduce the results of the paper "Statistical supervised meta-ensemble algorithm for data linkage". The vast majority of the code to reproduce the paper’s results was sourced from the paper’s GitHub repository https://github.com/ePBRN/Medical-Record-Linkage-Ensemble. [1]

### Citation to the Original Paper
K. Vo, J. Jitendra and L. Siaw-Teng, "Statistical supervised meta-ensemble algorithm for medical record linkage," Journal of Biomedical Informatics, 2019. 

### Link to the Original Paper’s Repository 
https://github.com/ePBRN/Medical-Record-Linkage-Ensemble

### Dependencies
All the code in this repository uses Python 3.10 with these prerequisite packages: numpy, pandas, sklearn, and recordlinkage. The following packages were used:
1. numpy 1.22.0
2. pandas 1.4.2
3. sklearn 1.0.2
4. recordlinkage v0.15
5. scipy 1.8.0
6. matplotlib 3.5.2
7. collections
8. math
9. statistics
10. IPython

## 1. Reproducing the Paper’s Results
### 1.1 Preprocessing the Datasets
The paper uses two datasets, including the freely extensible biomedical record linkage (FEBRL) datasets and the electronic practice-based research network (ePBRN) dataset.

The FEBRL dataset is an opensource dataset provided by the Python Record Linkage Toolkit library. The FEBRL dataset was developed with an error generator. [2] The authors provided code to use the Python Record Linkage Toolkit library and process the data. However, the regenerated FEBRL datasets are slightly different than the FEBRL datasets published on the author's GitHub repository. [1]  This is because the generation of the FEBRL is dependent on the version of Python Record Linkage Toolkit library used at the time.  When consulting with Jitendra Jonnagaddala, one of the paper's authors, it was stated that a reasonable explanation for this observed difference between the FEBRL datasets published on the authors' GitHub and the current regeneration of the datasets using the Python Record Linkage Toolkit library was due to changes in the library. The paper was published in 2019 and the most recent change to the library was committed on April 19, 2022. [2]

For the ePBRN dataset, the University of New South Wales ePBRN has been extracting clinical and administrative data from electronic health records (EHRs) for research purposes. The authors observed the medical linkage errors in the real-word Australian primary care facility and replicated these errors on the FEBRL datasets to produce the ePBRN dataset. We did not have access to the original source of data from the University of New South Wales ePBRN program. Ideally, the ePBRN_D_original.csv and ePBRN_F_original.csv files published on the author’s GitHub  [1] were thought to represent the originally FEBRL datasets that were modified to mimic the real-word errors observed in the Australian primary care facility. __However, in speaking with Jitendra Jonnagaddala, one of the paper's authors, it was stated that these files posted on the authors' GitHub are just examples and are not representative of the dataset used to produce the paper results.__
__

We used the example files (ePBRN_D_original.csv and ePBRN_F_original.csv) and the data processing code that were published on the authors’ GitHub to produce the ePBRN datasets ( ePBRN_D_dup.csv and ePBRN_F_dup.csv) that were fed to the models. __Since these datasets are not reflective of the datasets used in the study, it is not expected that these datasets will produce comparable results. We included this dataset as part of the study to evaluate how models performed when given a different dataset.__

```diff
+Action Item: Data download instruction
```
1. Create a folder called Data_to_produce_ePBRN_dataset within the root folder of where you will be downloading and run the python scripts
2. Download the ePBRN_D_original.csv and the ePBRN_F_original.csv from the paper's GitHub (https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/) and save them within the  Data_to_produce_ePBRN_dataset folder

```diff
+Action Item: Running script 
```
__Run the python file Preparing_FEBRL_and_ePBRN_Datasets.ipynb.__

__Inputs:__
1. ePBRN_D_original.csv – Stored in the Data_to_produce_ePBRN_dataset folder 
(Attained from the paper’s GitHub Data_to_produce_ePBRN_dataset/ePBRN_D_original.csv  [1]. This is not the actual dataset that the authors’ used to generate their results. Instead, it is an example of actual dataset)
2. ePBRN_F_original.csv – Stored in the Data_to_produce_ePBRN_dataset folder
(Attained from the paper’s GitHub Data_to_produce_ePBRN_dataset/ePBRN_F_original.csv [1]. This is not the actual dataset that the authors’ used to generate their results. Instead, it is an example of actual dataset)

__Outputs:__
1. febrl3_UNSW.csv
(As noted above this file does not exactually match the febrl3_UNSW.csv file published on the authors’ GitHub  [1]. This is likely due to using different variations of the Python Record Linkage Toolkit library)
2. febrl4_UNSW.csv
(As noted above this file does not exactually match the febrl4_UNSW.csv file published on the authors’ GitHub  [1]. This is likely due to using different variations of the Python Record Linkage Toolkit library)
3. ePBRN_D_dup.csv
(As noted above this file does not reflect the dataset used in by the authors to produce their results. As a result, the results derived from this dataset are expected to differ from the results noted in the paper.)
4. ePBRN_F_dup.csv
(As noted above this file does not reflect the dataset used in by the authors to produce their results. As a result, the results derived from this dataset are expected to differ from the results noted in the paper.)

### 1.2 Training and Evaluating: Reproduce the results from the paper (Table 4 and Table 6)
UNSW_Linkage.ipynb to reproduce the paper’s results. Specifically, Table 4 and Table 6 from the paper are recreated. As previously stated, the results derived from the FEBRL dataset are expected to be comparable to the results reported in the paper because the regenerated FEBRL datasets are similar FEBRL datasets published on the author’s GitHub (but not exactly the same). The regenerated ePBRN datasets are not representative of the ePBRN datasets used to produce the paper’s results. Thus, the results derived from the ePBRN dataset are expected to differ from the results noted in the paper

```diff
+Action Item: Running script 
```
__Run the python file UNSW_Linkage.ipynb.__

__Inputs:__
1. febrl3_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
2. febrl4_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
3. ePBRN_D_dup.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the Data_to_produce_ePBRN_dataset folder
4. ePBRN_F_dup.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the Data_to_produce_ePBRN_dataset folder

__Outputs:__

No files outputted

__Table of results:__
The script will produce the following tables of results. Note, the numerical values will change with every run dues to the inherent randomness of the method.

*FEBRL Regen. Blocking Results*

| Blocking Criterion  | Measure  | FEBRL Results (Mean of 10 Runs)  |
| -------------  | -------------  |  ------------- |
| Surname  | nc  | 170843 |
| Surname  | pc  | 66.500000 |
| Surname  | rr  | 99.658280 |
| Given name  | nc  | 154898 |
| Given name  | pc  | 65.740000 |
| Given name  | rr  | 99.690173 |
| Postcode  | nc | 53197 |
| Postcode  | pc | 84.380000 |
| Postcode  | rr | 99.893595 |
| All  | nc | 372073 |
| All  | pc | 97.880000 |
| All  | rr | 99.255780 |

*FEBRL Regen. Classification Results - Means*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 94.970645  | 99.683286 | 97.269678 | 273.9 |
| SVM-bag  | 96.263328  | 96.263328 | 97.928231 | 206.4 |
| NN  |  96.596002  | 99.650593 | 98.099327 | 189.0 |
| NN-bag  | 96.967586  | 99.636289 | 98.283779 | 170.3 |
| LR  | 86.656968  | 99.822231 | 92.768180 | 762.5 |
| LR-bag  |  87.628162  | 99.814058 | 93.318908 | 700.2 |
| Stack+Bag  |  97.821222  | 99.609726 | 98.707265 | 127.7 |

*FEBRL Regen. Classification Results - Standard deviation*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 0.290977  | 0.013707 | 0.149483 | 15.459625 |
| SVM-bag  | 	0.399282  | 0.009138 | 0.204158 | 20.784610 |
| NN  |  0.271229  | 0.006130 | 0.138707 | 14.071247 |
| NN-bag  | 0.130552  | 0.008173 | 0.065836 | 6.633250 |
| LR  | 1.462529  | 0.030376 | 0.814931 | 91.318125 |
| LR-bag  |  1.403998 | 0.029540 | 0.776850 | 86.348133 |
| Stack+Bag  |  0.197872  | 0.011004 | 0.097296 | 9.695360 |

*ePBRN Blocking Results*

| Blocking Criterion  | Measure  | FEBRL Results (Mean of 10 Runs)  |
| -------------  | -------------  |  ------------- |
| Surname  | nc  | 32785 |
| Surname  | pc  | 55.592967 |
| Surname  | rr  | 99.952446 |
| Given name  | nc  | 254696 |
| Given name  | pc  | 59.221848 |
| Given name  | rr  | 99.630571 |
| Postcode  | nc | 79556 |
| Postcode  | pc | 94.051627 |
| Postcode  | rr | 99.884606 |
| All  | nc | 363574 |
| All  | pc | 98.204265 |
| All  | rr | 99.472647 |

*ePBRN Classification Results - Means*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 33.962085  | 99.226667 | 50.598924 | 5089.2 |
| SVM-bag  | 38.986579  | 98.998095 | 55.936975 | 4096.4 |
| NN  | 69.698121  | 97.333333 | 81.229126 | 1180.9 |
| NN-bag  | 70.777419  | 97.333333 | 81.957394 | 1125.0 |
| LR  | 59.677290  | 97.600000 | 74.066237 | 1794.2 |
| LR-bag  |  60.345107  | 97.592381 | 74.575984 | 1746.8 |
| Stack+Bag  | 73.925839  | 97.287619 | 84.012545 | 972.0 |

*ePBRN Classification Results - Standard deviation*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 0.778565  | 0.017457 | 0.870004 | 15.459625 |
| SVM-bag  | 	0.843877  | 0.038285 | 0.866809 | 20.784610 |
| NN  |  0.347078  | 0.000000 | 0.235776 | 14.071247 |
| NN-bag  | 0.356078  | 0.000000 | 0.238962 | 6.633250 |
| LR  | 0.301553  | 0.000000 | 0.232245 | 91.318125 |
| LR-bag  |  0.355789 | 0.022857 | 0.267191 | 86.348133 |
| Stack+Bag  |  0.296642  | 0.015238 | 0.188759 | 13.747727 |

## 2. Additional Ablations:
### 2.1 Neural Network's Sensitivity to the Structure of Hidden Layers
This additional ablation file assesses the neural network models’ sensitivity to the structure of hidden layers. This analysis was performed on the base learners “NN” and “NN-bag”. Since “NN-bag” is a component of the ensemble model, the effect of the hidden layer changes was also monitored for the ensemble model. This analysis was performed using the FEBRL dataset.

```diff
+Action Item: Running script 
```
__Run the python file Ablation_Sensitivity_To_Hidden_Layer_Structure.__

__Inputs:__
1. febrl3_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
2. febrl4_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 

__Outputs:__
1. sensitivity_to_amount_of_training_data.jpeg – A graph of the sensitivity results 

### 2.2 Evaluating Varied Neural Network Configurations
Our second ablation is in the notebook Ablation_NN.ipynb.
Here, we evaluated the perfomance of competing Neural Network model configurations on the FEBRL Regen. (regenerated) dataset. Specifically, the following three model configuations were prepared:
1. NN with Logistic activation function (original study
used ’RELU’).
2. NN with ’ADAM’ solver (original study used ’lbfgs’).
3. NN with ’SGD’ (Stochastic Gradient Descent) solver.

All other configuration parameters were left unchanged except those identified above. These models were then benchmarked on the FEBRL Regn. dataset dataset over 10 runs and reported the following averaged results (along with their respective standard deviations).

```diff
+Action Item: Running script 
```
__Run the python file Ablation_NN.__


__Inputs:__
1. febrl3_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
2. febrl4_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 

__Outputs:__
1. A table of evaluations on the three competing custom NN models averaged over 10 runs. Means and standard deviations of the results have been reported. 


__Table of results:__
The script will produce the following tables of results. Note, the numerical values will change with every run dues to the inherent randomness of the method.

*NN Ablations - Custom Models Results*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| NN - Original  | 96.49 | 99.65 | 98.04 | 194.6 |
| NN - Log. Activation  | 98.68 | 99.36 | 99.016 | 96.6 |
| NN - Adam Solver |  98.78 | 99.10 | 98.94 | 104.4 |
| NN - SGD Solver  | 98.85 | 98.80 | 98.82 | 115.5 |

As shown above, the results of this ablation provide us with some exciting observations. It can be noted that all our custom configurations of the original NN model perfom significantly better than the one used in the original study. Moreover, they even perform better than every other model in the original study, including the ensemble model claimed by the authors as the best performer. Specifically, our NN model using Adam solver performs with the highest precision (98.85%) while Logistic Activation provides the smallest number of False Counts (96.6).

## 3. Supporting Analysis 
### 3.1 UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors – In the FEBRL_Class_Perf_When_Using_the_FEBRL_Dataset_Provided_on_Authors_GitHub Folder
Similar to UNSW_Linkage.ipynb to reproduce the paper’s results, the UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors file reproduces the results of the paper’s Table 4 and Table 6 corresponding the FEBRL dataset. However rather than regenerating the FEBRL dataset from the Python Record Linkage Toolkit library, it uses the FEBRL datasets published on the authors’ GitHub. [1] As stated above, the regenerated FEBRL datasets are slightly different than the FEBRL datasets published on the authors’ GitHub. It is expected that different datasets will lead to different results. Thus, to help eliminate this factor of variation in the results when attempting to reproduce the study, the FEBRL datasets published on the authors’ GitHub were used. Because these datasets are likely to be the most similar to the datasets used in the original study. 

```diff
+Action Item: Running script  
```
__Run the python file UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors.__

__Inputs:__
1. febrl3_UNSW_provided_by_authors.csv
(This file is equivalent to the febrl3_UNSW.csv file on the authors’ GitHub [1])
2. febrl4_UNSW_provided_by_authors.csv 
(This file is equivalent to the febrl4_UNSW.csv file on the authors’ GitHub [1])

__Outputs:__

No files outputted

__Table of results:__
The script will produce the following tables of results. Note, the numerical values will change with every run dues to the inherent randomness of the method.

*FEBRL Auth. Blocking Results*

| Blocking Criterion  | Measure  | FEBRL Results (Mean of 10 Runs)  |
| -------------  | -------------  |  ------------- |
| Surname  | nc  | 170843 |
| Surname  | pc  | 66.500000 |
| Surname  | rr  | 99.658280 |
| Given name  | nc  | 154898 |
| Given name  | pc  | 65.740000 |
| Given name  | rr  | 99.690173 |
| Postcode  | nc | 53197 |
| Postcode  | pc | 84.380000 |
| Postcode  | rr | 99.893595 |
| All  | nc | 372073 |
| All  | pc | 97.880000 |
| All  | rr | 99.255780 |

*FEBRL Auth. Classification Results - Means*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 95.096582  | 99.734369 | 97.359691 | 264.8 |
| SVM-bag  | 95.490281  | 99.734369 | 97.565787 | 243.6 |
| NN  |  93.027086  | 99.570903 | 96.187637 | 386.3 |
| NN-bag  | 93.075951  | 99.556600 | 96.207060 | 384.2 |
| LR  | 91.867286  | 99.754802 | 95.647753 | 444.4 |
| LR-bag  |  92.021193  | 99.750715 | 95.729193 | 435.7 |
| Stack+Bag  |  96.868159  | 99.495300 | 98.163815 | 182.2 |

*FEBRL Auth. Classification Results - Standard deviation*
| Model  | pr(%)  | re(%)  | fs(%)  | fc  |
| ------ | -----  | ------ | ------ | --- |
| SVM  | 	0.465962  | 0.000000 | 0.244357 | 25.179357 |
| SVM-bag  | 	0.389443  | 0.000000 | 0.203285 | 20.832667 |
| NN  |  0.257596  | 0.015827 | 0.136244 | 14.352700 |
| NN-bag  | 0.262914  | 0.015959 | 0.137260 | 14.456832 |
| LR  | 0.578806  | 0.009138 | 0.312181 | 33.346664 |
| LR-bag  |  0.586991 | 0.008173 | 0.315323 | 33.615473 |
| Stack+Bag  |  0.366441  | 0.013084 | 0.191381 | 19.339080 |


## Works Cited
[1] 	K. Vo, J. Jitendra and L. Siaw-Teng, "Statistical supervised meta-ensemble algorithm for medical record linkage," Journal of Biomedical Informatics, 2019. 

[2] 	J. de Bruin, "RecordLinkage: powerful and modular Python record linkage toolkit," 19 April 2022. [Online]. Available: https://github.com/J535D165/recordlinkage.

[3] 	K. Vo, J. Jonnagaddala and S.-T. Liaw, "Medical-Record-Linkage-Ensemble," 16 February 2019. [Online]. Available: https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/.

[4] 	P. Christen, "Febrl - An open source data cleaning, deduplication and record linkage system with a graphical user interface," Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, p. 1065–1068, August 2008. 
