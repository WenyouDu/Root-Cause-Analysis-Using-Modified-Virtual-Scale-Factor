# Root-Cause-Analysis-Using-Modified-Virtual-Scale-Factor
An incremental analysis method was proposed in this work to improve the interpretability of data-driven fault detection models. Without additional training, the contribution of each feature to the model decision could be quantified by using only the anomaly detection model and the current sample.
## Authors
Wenyou Du, Yutong Meng
## About TEP_DATA
1. This dataset is consistent with the dataset introduced in the textbook written by Jiang Haotian.
2. 
3. Files ending with "te" correspond to test sets, while files without the suffix "te" are training sets.
4. 
5. Files numbered d01–d021 contain data for 21 different faults, and d00 contains normal operating data.
6. 
7. For training data: d00.dat consists of 500 normal samples; d01.dat to d021.dat each contain 480 samples, with faults injected starting from the 80th sample.
8. 
9. All test datasets include 960 samples. For fault-related test data, faults are introduced from the 160th sample onward.
10. 
11. Special note: In d00.dat, rows represent variables and columns represent samples. For all other data files, columns denote variables and rows denote samples.
12. 
13. Refer to the READ_TEP.ipynb script in the folder for data loading implementations.
14. 

## Status
This work is currently submitted to the Journal of Process Control.

## Usage
All result in the paper are contained in the four notebook files. Just install related python environment and run the .ipynb files.

## E-mail adress
wen-you.du@sau.edu.cn
