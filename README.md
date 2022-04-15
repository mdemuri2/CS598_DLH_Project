# CS598_DLH_Project
Deep Learning Health Care Project: Statistical supervised meta-ensemble algorithm for medical record linkage

This repository contains the full code for the CS 598 Deep Learning for Healthcare project. The main goal of the project was to reproduce the results of the paper "Statistical supervised meta-ensemble algorithm for data linkage". The vast majority of the code to reproduce the paper’s results was sourced from the paper’s GitHub repository https://github.com/ePBRN/Medical-Record-Linkage-Ensemble. [1]

All the code in this repository uses Python 3.10 with these prerequisite packages: numpy, pandas, sklearn, and recordlinkage. 

## Step 1: Prepare the datasets
The paper uses two datasets, including the freely extensible biomedical record linkage (FEBRL) datasets and the electronic practice-based research network (ePBRN) dataset.  
 
The FEBRL dataset is an opensource dataset provided by the Python Record Linkage Toolkit library. The FEBRL dataset was developed with an error generator. [2] 
 
For the ePBRN dataset, the University of New South Wales ePBRN has been extracting clinical and administrative data from electronic health records (EHRs) for research purposes. The authors observed the medical linkage errors in the real-word Australian primary care facility and replicated these errors on the FEBRL datasets to produce the ePBRN dataset. We did not have access to the original source of data from the University of New South Wales ePBRN program.  However, to produce the ePBRN dataset the researchers have provided original FEBRL dataset, and the code needed to change the FEBRL dataset so that it mimicked the real-word errors in accordance with the errors observed in the Australian primary care facility.  We utilized the data and code on the paper’s GitHub repository to replicate the ePBRN for our study. [1]

Run Preparing_FEBRL_and_ePBRN_Datasets.ipynb to create the FEBRL dataset and the ePBRN dataset. The FEBRL dataset will be stored in two files febrl3_UNSW.csv and febrl3_UNSW.csv. The original FEBRL datasets for the ePBRN datasets are contained in 2 files: / Data_to_produce_ePBRN_dataset/ePBRN_D_original.csv and / Data_to_produce_ePBRN_dataset/ePBRN_F_original.csv. Running the Preparing_FEBRL_and_ePBRN_Datasets.ipynb takes the ePBRN_D_original.csv and ePBRN_F_dup.csv files and replicates the errors observed in the real-world Australian primary care facility within them. The ePBRN datasets are stored in two files ePBRN_D_dup.csv and ePBRN_F_dup.csv

## Step 2: Reproduce the results from the paper (Table 4 and Table 6)
Run UNSW_Linkage.ipynb to reproduce the paper’s results. Specifically, Table 4 and Table 6 from the paper are recreated.

## Works Cited

[1] 	K. Vo, J. Jonnagaddala and S.-T. Liaw, "Statistical supervised meta-ensemble algorithm for medical record linkage," Journal of Biomedical Informatics, vol. 95, no. 1532-0464, 31 May 2019. 

[2] 	P. Christen, "Febrl - An open source data cleaning, deduplication and record linkage system with a graphical user interface," Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, p. 1065–1068, August 2008. 
