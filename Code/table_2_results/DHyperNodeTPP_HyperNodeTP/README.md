# scalable_hyperedge_forecasting

## Dataset
| Datasets | ID  | 
|:---------|:---:|
| enron    |  0  | 
| eumail   |  1  | 
| hepth    |  2  |
| arxiv    |  3  | 
| twitter  |  4  | 



* For directed models, DHyperNodeTPP
    * python node_event_hgcn_directed.py --dataset dataset_id --gpu 0  --freq 1000 --fileID logfilename --type d --seed 0  --model_suffix string --alpha 0.3 --epoch num_of_epochs
* For undirected models, HyperNodeTPP
    * python node_event_hgcn_directed.py --dataset dataset_id --gpu 0  --freq 1000 --fileID logfilename --mask --type u --seed 0 --model_suffix string --alpha 0.3 --epoch num_of_epochs