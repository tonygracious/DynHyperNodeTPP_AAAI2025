# Hyper-SAGNN, CATSETMAT and OUR Model for Static Datasets

## Dataset
| Datasets | ID  | 
|:---------|:---:|
| enron    |  0  | 
| hepth    |  2  |
| arxiv    |  3  | 
| Twitter  |  4  | 
| iAF1260b |  5  | 
| iJO1366  |  6  | 
| uspto    |  7  | 

* For Hyper-SAGNN-
  * cd Hyper-SAGNN-CAT
  * python main.py --datasetpath ../../directed_datasets/ --filename DATASET_NAME --dimensions 64 --bs 32 --epochs 100 --gpu 2 -d --patience 10 --model hsgnn --n_run 3 

* For CATSETMAT
  * cd Hyper-SAGNN-CAT
  * python main.py --datasetpath ../../directed_datasets/ --filename DATASET_NAME --dimensions 64 --bs 32 --epochs 100 --gpu 2 --patience 10 --model cat --cat_model x --n_run 3 

* OUR(Static Datasets)
  * cd Hyper-SAGNN-CAT
  * python main.py --datasetpath ../../directed_datasets/ --filename DATASET_NAME --dimensions 64 --bs 32 --epochs 1 --gpu 2 --patience 10 --model dir_cat --cat_model x --n_run 3 

