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

<span style="color: blue">__Action Item:__</span>

## Step 2: Reproduce the results from the paper (Table 4 and Table 6)
Run UNSW_Linkage.ipynb to reproduce the paper’s results. Specifically, Table 4 and Table 6 from the paper are recreated.

Note: please ignore the .ipynb_checkpoints folder

## Works Cited

[1] 	K. Vo, J. Jonnagaddala and S.-T. Liaw, "Statistical supervised meta-ensemble algorithm for medical record linkage," Journal of Biomedical Informatics, vol. 95, no. 1532-0464, 31 May 2019. 

[2] 	P. Christen, "Febrl - An open source data cleaning, deduplication and record linkage system with a graphical user interface," Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, p. 1065–1068, August 2008. 
