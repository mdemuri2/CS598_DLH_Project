# CS598_DLH_Project
#### Maria DeMuri | NetID: mdemuri2 | mdemuri2@illinois.edu   
#### Salman Yousaf | NetID: syousaf2 | syousaf2@illinois.edu
## Deep Learning Health Care Project: Statistical supervised meta-ensemble algorithm for medical record linkage

This repository contains the full code for the CS 598 Deep Learning for Healthcare project. The main goal of the project was to reproduce the results of the paper "Statistical supervised meta-ensemble algorithm for data linkage". The vast majority of the code to reproduce the paper’s results was sourced from the paper’s GitHub repository https://github.com/ePBRN/Medical-Record-Linkage-Ensemble. [1]

All the code in this repository uses Python 3.10 with these prerequisite packages: numpy, pandas, sklearn, and recordlinkage. The following versions of the packages were used:
1. numpy 1.22.0
2. pandas 1.4.2
3. sklearn 1.0.2
4. recordlinkage v0.15

## 1. Reproducing the Paper’s Results
### 1.1 Prepare the Datasets
The paper uses two datasets, including the freely extensible biomedical record linkage (FEBRL) datasets and the electronic practice-based research network (ePBRN) dataset.

The FEBRL dataset is an opensource dataset provided by the Python Record Linkage Toolkit library. The FEBRL dataset was developed with an error generator. [2] The authors provided code to use the Python Record Linkage Toolkit library and process the data. However, the regenerated FEBRL datasets are slightly different than the FEBRL datasets published on the author's GitHub repository. [1]  This is because the generation of the FEBRL is dependent on the version of Python Record Linkage Toolkit library used at the time.  When consulting with Jitendra Jonnagaddala, one of the paper's authors, it was stated that a reasonable explanation for this observed difference between the FEBRL datasets published on the authors' GitHub and the current regeneration of the datasets using the Python Record Linkage Toolkit library was due to changes in the library. The paper was published in 2019 and the most recent change to the library was committed on April 19, 2022. [2]

For the ePBRN dataset, the University of New South Wales ePBRN has been extracting clinical and administrative data from electronic health records (EHRs) for research purposes. The authors observed the medical linkage errors in the real-word Australian primary care facility and replicated these errors on the FEBRL datasets to produce the ePBRN dataset. We did not have access to the original source of data from the University of New South Wales ePBRN program. Ideally, the ePBRN_D_original.csv and ePBRN_F_original.csv files published on the author’s GitHub  [1] were thought to represent the originally FEBRL datasets that were modified to mimic the real-word errors observed in the Australian primary care facility. __However, in speaking with Jitendra Jonnagaddala, one of the paper's authors, it was stated that these files posted on the authors' GitHub are just examples and are not representative of the dataset used to produce the paper results.__
__

We used the example files (ePBRN_D_original.csv and ePBRN_F_original.csv) and the data processing code that were published on the authors’ GitHub to produce the ePBRN datasets ( ePBRN_D_dup.csv and ePBRN_F_dup.csv) that were fed to the models. __Since these datasets are not reflective of the datasets used in the study, it is not expected that these datasets will produce comparable results. We included this dataset as part of the study to evaluate how models performed when given a different dataset.__
```diff
+Action Item:
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

## 1.2 Reproduce the results from the paper (Table 4 and Table 6)
UNSW_Linkage.ipynb to reproduce the paper’s results. Specifically, Table 4 and Table 6 from the paper are recreated. As previously stated, the results derived from the FEBRL dataset are expected to be comparable to the results reported in the paper because the regenerated FEBRL datasets are similar FEBRL datasets published on the author’s GitHub (but not exactly the same). The regenerated ePBRN datasets are not representative of the ePBRN datasets used to produce the paper’s results. Thus, the results derived from the ePBRN dataset are expected to differ from the results noted in the paper
```diff
+Action Item:
```
__Run the python file UNSW_Linkage.ipynb.__

__Inputs:__
1. febrl3_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
2. febrl4_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
3. ePBRN_D_dup.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the Data_to_produce_ePBRN_dataset folder
4. ePBRN_F_dup.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the Data_to_produce_ePBRN_dataset folder

__Outputs:__

No files outputted

## Additional Ablations:
This additional ablation file assesses the neural network models’ sensitivity to the structure of hidden layers. This analysis was performed on the base learners “NN” and “NN-bag”. Since “NN-bag” is a component of the ensemble model, the effect of the hidden layer changes was also monitored for the ensemble model. This analysis was performed using the FEBRL dataset.

```diff
+Action Item:
```
__Run the python file Ablation_Sensitivity_To_Hidden_Layer_Structure.__

__Inputs:__
1. febrl3_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 
2. febrl4_UNSW.csv - Produced by the Preparing_FEBRL_and_ePBRN_Datasets.ipynb file and stored in the root folder 

__Outputs:__
1. sensitivity_to_amount_of_training_data.jpeg – A graph of the sensitivity results 

## Supporting Analysis 
### 3.1 UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors – In the Supporting_Analysis Folder
Similar to UNSW_Linkage.ipynb to reproduce the paper’s results, the UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors file reproduces the results of the paper’s Table 4 and Table 6 corresponding the FEBRL dataset. However rather than regenerating the FEBRL dataset from the Python Record Linkage Toolkit library, it uses the FEBRL datasets published on the authors’ GitHub. [1] As stated above, the regenerated FEBRL datasets are slightly different than the FEBRL datasets published on the authors’ GitHub. It is expected that different datasets will lead to different results. Thus, to help eliminate this factor of variation in the results when attempting to reproduce the study, the FEBRL datasets published on the authors’ GitHub were used. Because these datasets are likely to be the most similar to the datasets used in the original study. 

```diff
+Action Item:
```
__Run the python file UNSW_Linkage_Original_FEBRL_Provided_By_The_Authors.__

__Inputs:__
1. febrl3_UNSW_provided_by_authors.csv
(This file is equivalent to the febrl3_UNSW.csv file on the authors’ GitHub [1])
2. febrl4_UNSW_provided_by_authors.csv 
(This file is equivalent to the febrl4_UNSW.csv file on the authors’ GitHub [1])

__Outputs:__

No files outputted



## Works Cited
[1] 	K. Vo, J. Jitendra and L. Siaw-Teng, "Statistical supervised meta-ensemble algorithm for medical record linkage," Journal of Biomedical Informatics, 2019. 

[2] 	J. de Bruin, "RecordLinkage: powerful and modular Python record linkage toolkit," 19 April 2022. [Online]. Available: https://github.com/J535D165/recordlinkage.

[3] 	K. Vo, J. Jonnagaddala and S.-T. Liaw, "Medical-Record-Linkage-Ensemble," 16 February 2019. [Online]. Available: https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/.

[4] 	P. Christen, "Febrl - An open source data cleaning, deduplication and record linkage system with a graphical user interface," Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, p. 1065–1068, August 2008. 
