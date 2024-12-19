import sys
sys.path.append('..')
import torch
import torch.nn as nn
import os
import time
import argparse
import warnings
from Modules import *
from utils import *
from Dataloader_directed import HyperDiEdgeDataset , size_distribution
from preprocess import process_hyperedges, padding ,processStaticData
import matplotlib as mpl
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
mpl.use("Agg")
import multiprocessing

cpu_num = multiprocessing.cpu_count()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")

def parse_args():
    # Parses the arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--datasetpath',default = '../../directed_datasets', help='Enter the common path to datasets folder')
    parser.add_argument('--filename', default = 'enron',
     choices=['iAF1260b','iJO1366','uspto','arxiv_25' ,'enron', 'eumail' , 'hepth','twitter'] ,help = 'Enter the dataset folder to use')
    parser.add_argument('--dimensions', type=int, default=64 ,
                        help='Number of dimensions. Default is 64.')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('-d', '--diag', type=str, default='False',
                        help='Use the diag mask or not')
    parser.add_argument('--gpu'  , default= '3', help= 'Cuda device number')
    parser.add_argument('--patience',type = int, default= 10, help = 'Early Stopping Limit')
    parser.add_argument('--model' , type = str , choices=['hsgnn','cat','dir_cat'], default='dir_cat')
    parser.add_argument('--cat_model', type =str , choices=['x','sx','sxs'], default='x')
    parser.add_argument('--split', type =str, default= 1, help ='Train Test Split to use in Non temporal Data')
    parser.add_argument('--n_run', type = int, default = 1,help = 'Number of time running experiments')
    parser.add_argument('--bs', type= int, default= 32 , help = 'batch size')
    args = parser.parse_args()
    print(args)
    return args


def train_batch_hyperedge(model, loss_func, batch_hyperedges):
    neg_hyperedges = negative_sampling_hyperedge_directed(batch_hyperedges.tolist()\
         , max_nodes=(train_num_nodes -1 , train_num_nodes -1), p = p ,Neg_per_Edge= neg_num, seed= None)
    
    # HyperSAGNN
    if args.model == 'hsgnn':
        # HyperSAGNN model is solving easier problem which is identifying nodes in the hyperedges.
        # It does not give any information between nodes in hyperedges.

        pos_x , pos_max_len = process_hyperedges(batch_hyperedges)
        neg_x, neg_max_len = process_hyperedges(neg_hyperedges)
        max_len = max(pos_max_len, neg_max_len)

        pos_x = padding(pos_x, max = max_len , pad_idx= 0)
        neg_x = padding(neg_x, max = max_len, pad_idx= 0)

        x = np.concatenate((pos_x , neg_x), axis = 0)
        y = np.concatenate((np.ones(pos_x.shape[0]) , np.zeros(neg_x.shape[0])))
        
        x, y = torch.from_numpy(x).to(device) , torch.from_numpy(y).to(device)
        index = torch.randperm(len(x))
        x, y = x[index], y[index]
        y = y.view(-1,1).float()
        # forward
        pred = model(x)
        loss = loss_func(pred,y) 
    
    # CATSETMAT and Our Model
    else:
        pos_x1 , pos_x2 , pos_max_len1 , pos_max_len2 = process_hyperedges(batch_hyperedges, model_name= args.model)
        neg_x1 , neg_x2 , neg_max_len1 , neg_max_len2 = process_hyperedges(neg_hyperedges, model_name= args.model)
        max_len1 = max(pos_max_len1, neg_max_len1)
        max_len2 = max(pos_max_len2, neg_max_len2)
        pos_x1 = padding(pos_x1, max = max_len1 , pad_idx= 0)
        neg_x1 = padding(neg_x1, max = max_len1, pad_idx= 0) 
        pos_x2 = padding(pos_x2, max = max_len2 , pad_idx= 0)
        neg_x2 = padding(neg_x2, max = max_len2, pad_idx= 0)

        x1 = np.concatenate((pos_x1 , neg_x1), axis = 0)
        x2 = np.concatenate((pos_x2 , neg_x2), axis = 0)

        y = np.concatenate((np.ones(pos_x1.shape[0]) , np.zeros(neg_x1.shape[0])))
        
        x1, x2, y = torch.from_numpy(x1).to(device),torch.from_numpy(x2).to(device)  , torch.from_numpy(y).to(device)
        index = torch.randperm(len(x1))
        x1, x2, y = x1[index], x2[index], y[index]
        y = y.view(-1,1).float()
        pred, _ = model(x1 , x2)
        loss = loss_func(pred,y) 
    return pred, y, loss



def train_epoch(args, model, loss_func, hyperedges, optimizer, batch_size):
    """
        1 Training Epoch
    """
    model.train()
    hyperedges = np.array(hyperedges,dtype=object)
    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []
    batch_num = math.ceil(len(hyperedges) / batch_size)
    for edge_id in tqdm(DataLoader(range(len(hyperedges)), batch_size, shuffle=True)):
        optimizer.zero_grad()
        edge_id = edge_id.numpy()
        batch_hyperedges = hyperedges[edge_id]
        pred, batch_y, loss_bce = train_batch_hyperedge(model, loss_func, batch_hyperedges)
        loss = loss_bce 
        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bce_total_loss += loss_bce.item()

    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    auc1, auc2 = roc_auc_cuda(y, pred)
    return bce_total_loss / batch_num,  np.mean(acc_list), auc1, auc2


def eval_epoch(args, model, loss_func, validation_hyperedges, batch_size, seed):
    ''' Epoch operation in evaluation phase '''
    bce_total_loss = 0
    model.eval()
    with torch.no_grad():
        validation_hyperedges = np.array(validation_hyperedges,dtype=object)
        pred, label, ranks = [], [] ,[]
        batch_num = math.ceil(len(validation_hyperedges)/batch_size)
        for edge_id in tqdm(DataLoader(range(len(validation_hyperedges)), batch_size, shuffle=False)):
            edge_id = edge_id.numpy()
            batch_hyperedges = validation_hyperedges[edge_id]
            # prepare data
            neg_hyperedges = negative_sampling_hyperedge_directed(batch_hyperedges.tolist()\
                    , max_nodes=(num_nodes-1 , num_nodes-1), p = p , Neg_per_Edge= neg_num, seed = seed)

            if args.model == 'hsgnn':
                pos_x , pos_max_len = process_hyperedges(batch_hyperedges)
                neg_x, neg_max_len = process_hyperedges(neg_hyperedges)
                max_len = max(pos_max_len, neg_max_len)

                pos_x = padding(pos_x, max = max_len , pad_idx= 0)
                neg_x = padding(neg_x, max = max_len, pad_idx= 0)

                pos_x , neg_x = torch.from_numpy(pos_x).to(device) , torch.from_numpy(neg_x).to(device)
                pred_pos =  model(pos_x)
                pred_neg = model(neg_x)
                y = np.concatenate((np.ones(pos_x.shape[0]) , np.zeros(neg_x.shape[0])))
                y =torch.from_numpy(y).to(device)
                y = y.view(-1,1).float()
            else:
                pos_x1 , pos_x2 , pos_max_len1 , pos_max_len2 = process_hyperedges(batch_hyperedges, model_name= args.model)
                neg_x1 , neg_x2 , neg_max_len1 , neg_max_len2 = process_hyperedges(neg_hyperedges, model_name= args.model)
                max_len1 = max(pos_max_len1, neg_max_len1)
                max_len2 = max(pos_max_len2, neg_max_len2)
                pos_x1 = padding(pos_x1, max = max_len1 , pad_idx= 0)
                neg_x1 = padding(neg_x1, max = max_len1, pad_idx= 0) 
                pos_x2 = padding(pos_x2, max = max_len2 , pad_idx= 0)
                neg_x2 = padding(neg_x2, max = max_len2, pad_idx= 0)

                pos_x1 , neg_x1 = torch.from_numpy(pos_x1).to(device) , torch.from_numpy(neg_x1).to(device)
                pos_x2 , neg_x2 = torch.from_numpy(pos_x2).to(device) , torch.from_numpy(neg_x2).to(device)
                
                pred_pos,_ =  model(pos_x1 , pos_x2)
                pred_neg,_ = model(neg_x1 , neg_x2)
                y = np.concatenate((np.ones(pos_x1.shape[0]) , np.zeros(neg_x1.shape[0])))
                y =torch.from_numpy(y).to(device)
                y = y.view(-1,1).float()

            ranks.extend(getRank(pred_pos.cpu().numpy() , pred_neg.cpu().numpy().reshape(-1,20) , 0))
            
            pred_batch = torch.concat([pred_pos , pred_neg])
           

            pred.append(pred_batch)
            label.append(y)
            
            loss = loss_func(pred_batch,y)
            bce_total_loss += loss.item()
        
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        
        acc = accuracy(pred, label)
        mrr = MRR(ranks)
        auc1, auc2 = roc_auc_cuda(label, pred)
    
    return bce_total_loss / batch_num, acc, auc1, auc2 , mrr


def train(args, model, loss, training_data, validation_data,test_data, optimizer, epochs, batch_size):
    
    es_limit = args.patience
    pre_val_auc = -1e6
    best_epoch = -1
    for epoch_i in range(epochs):
        print('[ Epoch', epoch_i, 'of', epochs, ']')
        
        start = time.time()
        
        bce_loss, train_accu, auc1, auc2 = train_epoch(
            args, model, loss, training_data, optimizer, batch_size)
        print('  - (Training)   bce: {bce_loss: 7.4f}, '
              ' acc: {accu:3.3f} %, auc: {auc1:3.3f}, aupr: {auc2:3.3f}, '
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=bce_loss,
            accu=100 *
                 train_accu,
            auc1=auc1,
            auc2=auc2,
            elapse=(time.time() - start)))
        
        start = time.time()
        valid_bce_loss, valid_accu, valid_auc1, valid_auc2,val_mrr = eval_epoch(args, model, loss, validation_data, batch_size, seed = 0)
        print('  - (Validation-hyper) bce: {bce_loss: 7.4f}'
              '  acc: {accu:3.3f} %,'
              ' auc: {auc1:3.3f}, aupr: {auc2:3.3f}, mrr: {mrr:3.3f} '
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=valid_bce_loss,
            accu=100 *
                 valid_accu,
            auc1=valid_auc1,
            auc2=valid_auc2,
            mrr = val_mrr,
            elapse=(time.time() - start)))
        
        # Early Stopping 
        if pre_val_auc < valid_auc1:
            es_limit = args.patience
            pre_val_auc = valid_auc1
            best_epoch = epoch_i
            print('saving the model at epoch :' ,epoch_i)
            if args.model == 'hsgnn':
                torch.save(model.state_dict(), f'./best_model/{args.model}_{args.filename}_main2.pth')
            else:
                torch.save(model.state_dict(), f'./best_model/{args.model}_{args.filename}_{args.cat_model}.pth')

        else:
            es_limit -=1

        if es_limit == 0:
            print('Stopping Training at Epoch:',epoch_i)
            break

    print(f'Loading the model at best epoch {best_epoch}')
    if args.model == 'hsgnn':
        model.load_state_dict(torch.load(f'./best_model/{args.model}_{args.filename}_main2.pth'))
    else:
        model.load_state_dict(torch.load(f'./best_model/{args.model}_{args.filename}_{args.cat_model}.pth'))

    results = {
        'val_auc1':[], 'val_auc2':[],'val_mrr':[],
        'test_auc1':[], 'test_auc2':[],'test_mrr':[],
    }

    # Evaluating the model performance against the negative sampling at seed in range 0 to 5.
    for seed_i in range(6):
        print('Validating and Testing at seed ', seed_i)
        start = time.time()
        valid_bce_loss, valid_accu, valid_auc1, valid_auc2,val_mrr = eval_epoch(args, model, loss, validation_data, batch_size,seed_i)
        print('  - (Validation-hyper) bce: {bce_loss: 7.4f}'
                '  acc: {accu:3.3f} %,'
                ' auc: {auc1:3.3f}, aupr: {auc2:3.3f}, mrr: {mrr:3.3f} '
                'elapse: {elapse:3.3f} s'.format(
            bce_loss=valid_bce_loss,
            accu=100 *
                    valid_accu,
            auc1=valid_auc1,
            auc2=valid_auc2,
            mrr = val_mrr,
            elapse=(time.time() - start)))

        results['val_auc1'].append(valid_auc1)
        results['val_auc2'].append(valid_auc2)
        results['val_mrr'].append(val_mrr)
        
        start = time.time()
        test_bce_loss, test_accu, test_auc1, test_auc2, test_mrr = eval_epoch(args, model, loss, test_data, batch_size,seed_i)
        print('  - (test-hyper) bce: {bce_loss: 7.4f}'
                '  acc: {accu:3.3f} %,'
                ' auc: {auc1:3.3f}, aupr: {auc2:3.3f}, mrr:{mrr:3.3f} '
                'elapse: {elapse:3.3f} s'.format(
            bce_loss=test_bce_loss,
            accu=100 *
                    test_accu,
            auc1=test_auc1,
            auc2=test_auc2,
            mrr = test_mrr,
            elapse=(time.time() - start)))  
        results['test_auc1'].append(test_auc1)
        results['test_auc2'].append(test_auc2)
        results['test_mrr'].append(test_mrr) 
    

    results_for_n_exp['val_auc1'].append(np.mean(results['val_auc1']))
    results_for_n_exp['val_auc2'].append(np.mean(results['val_auc2']))
    results_for_n_exp['val_mrr'].append(np.mean(results['val_mrr']))
    results_for_n_exp['test_auc1'].append(np.mean(results['test_auc1']))
    results_for_n_exp['test_auc2'].append(np.mean(results['test_auc2']))
    results_for_n_exp['test_mrr'].append(np.mean(results['test_mrr']))

    print('Evaluation Stats For Current Experiment-')
    print('Validation')
    print(f"AUC:  mean {np.mean(results['val_auc1'])}   std {np.std(results['val_auc1'])}")
    print(f"AUPR:  mean {np.mean(results['val_auc2'])}   std {np.std(results['val_auc2'])}")
    print(f"MRR:  mean {np.mean(results['val_mrr'])}   std {np.std(results['val_mrr'])}")

    print('Testing')
    print(f"AUC:  mean {np.mean(results['test_auc1'])}   std {np.std(results['test_auc1'])}")
    print(f"AUPR:  mean {np.mean(results['test_auc2'])}   std {np.std(results['test_auc2'])}")
    print(f"MRR:  mean {np.mean(results['test_mrr'])}   std {np.std(results['test_mrr'])}")

args = parse_args()
neg_num = 20
batch_size = args.bs
bottle_neck = args.dimensions
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
# device = 'cpu'
temporal_data = ['arxiv_25' ,'enron', 'eumail' , 'hepth','twitter']
nontemporal_data = ['dblp','iAF1260b','iJO1366','reverb','uspto']

if args.filename in temporal_data :
    path = os.path.join(args.datasetpath, args.filename)
    # print(args.datasetpath , args.filename)
    data = HyperDiEdgeDataset(path)
    p = [data.p_right , data.p_left]
    num_nodes = data.n_nodes

    length = len(data.hyperedges)
    train_size , val_size , test_size = int(length * 0.5) , int(length * 0.25) , int(length * 0.25) 
    train_hyperedges = data.hyperedges[0 : train_size+1]
    val_hyperedges = data.hyperedges[train_size+1 : train_size + val_size + 1]
    test_hyperedges =data.hyperedges[train_size + val_size + 1 : ]

elif args.filename in nontemporal_data:
    print('Static Dataset :' , args.filename)
    path = os.path.join(args.datasetpath, args.filename)
    num_nodes, train_hyperedges , _ , val_test_hyperedges, _  =\
        processStaticData(path, args.split)
    val_hyperedges, test_hyperedges = train_test_split(val_test_hyperedges,test_size=0.5 , random_state = 42)
    
    size_right_left_nodes = np.array([[len(hyperedge[0][0]) , len(hyperedge[0][1])] for hyperedge in train_hyperedges]).T
    p_right = size_distribution(size_right_left_nodes[0])
    p_left = size_distribution(size_right_left_nodes[1])
    p = [p_right, p_left]
else:
    print('Dataset %s not found'%args.filename)

train_num_nodes = np.max([np.max(hyperedge[0][0] + hyperedge[0][1]) for hyperedge in train_hyperedges])

num_nodes +=1 

results_for_n_exp = {
        'val_auc1':[], 'val_auc2':[],'val_mrr':[],
        'test_auc1':[], 'test_auc2':[],'test_mrr':[],
}

setdevice(args.gpu)
print('Number of time the experiments will be running :' , args.n_run)
for i in range(args.n_run):
    print('*' * 20)
    print('Experiment Number : ', i)
    if args.model == 'hsgnn':
        print('HYPERSGNN MODEL')
        classifier_model = Classifier(
            n_head=2,
            d_model=args.dimensions,
            d_k=16,
            d_v=16,
            num_nodes = num_nodes,
            pad_idx = 0,
            emb_size = args.dimensions,
            diag_mask=args.diag,
            bottle_neck=bottle_neck,
            device = device).to(device)
    else:
        if args.model == 'cat':
            print('CATSETMAT MODEL')
            classifier_model = CatClassifier(
                n_head=2,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                num_nodes = num_nodes,
                pad_idx = 0,
                emb_size = args.dimensions,
                diag_mask=args.diag,
                bottle_neck=bottle_neck,
                cross_attn_type = args.cat_model).to(device)

        elif args.model == 'dir_cat':
            print('DIRECTED CATSETMAT MODEL')
            classifier_model = DirectedClassifier(
                n_head=2,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                num_nodes = num_nodes,
                pad_idx = 0,
                emb_size = args.dimensions,
                diag_mask=args.diag,
                bottle_neck=bottle_neck,
                softplus_layer = False,
                cross_attn_type = args.cat_model).to(device)
        else:
            raise Exception("Model%s Not Found!"%args.model)
    loss = F.binary_cross_entropy

    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)

    train(args, classifier_model,
        loss= loss,
        training_data=train_hyperedges,
        validation_data=val_hyperedges,
        test_data = test_hyperedges,
        optimizer= optimizer, epochs=args.epochs, batch_size=batch_size)


print('*' * 20)
print(f'Evaluation Stats Over {args.n_run} Experiments-')
print('Validation')
print(f"AUC:  mean {np.mean(results_for_n_exp['val_auc1'])}   std {np.std(results_for_n_exp['val_auc1'])}")
print(f"AUPR:  mean {np.mean(results_for_n_exp['val_auc2'])}   std {np.std(results_for_n_exp['val_auc2'])}")
print(f"MRR:  mean {np.mean(results_for_n_exp['val_mrr'])}   std {np.std(results_for_n_exp['val_mrr'])}")

print('Testing')
print(f"AUC:  mean {np.mean(results_for_n_exp['test_auc1'])}   std {np.std(results_for_n_exp['test_auc1'])}")
print(f"AUPR:  mean {np.mean(results_for_n_exp['test_auc2'])}   std {np.std(results_for_n_exp['test_auc2'])}")
print(f"MRR:  mean {np.mean(results_for_n_exp['test_mrr'])}   std {np.std(results_for_n_exp['test_mrr'])}")


