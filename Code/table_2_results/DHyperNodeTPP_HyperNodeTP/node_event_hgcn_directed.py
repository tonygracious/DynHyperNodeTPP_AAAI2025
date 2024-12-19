import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
import logging
import pandas as pd
import random
import pickle

from DataLoader.Dataloader_directed import HyperBiEdgeDataset, HyperDiEdgeDataset
from Utils.utils import batching_data, computeAucRocScore, save_predictions
from Utils.history import NeighborFinder
from pathlib import Path
from sklearn import metrics

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='continous embed')
    parser.add_argument('--fileID', type=str, default='enron_debugging')
    parser.add_argument('--dataset', type=int, default='0')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--mask',  default=False, action='store_true')
    parser.add_argument('--type', type=str, default='u')
    parser.add_argument('--neg_e', type=int, default=20)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--freq', type=float, default=1000)
    parser.add_argument('--g_type',type=str,default='directed')
    parser.add_argument('--epoch',type=int, default= 10)
    parser.add_argument("--use_wandb" , action="store_true")
    parser.add_argument('--alpha', type=float, default= 0.3, help='Alpha Hyperparameter In Connectivity Model')
    parser.add_argument("--model_suffix", type=str , default="JohnDoe", help ='Suffix to add to model before saving it')
    parser.add_argument("--use_last_checkpoint", default=False, action= 'store_true', help ='use last checkpoint')
    parser.add_argument("--start_epoch", type=int ,default= 0, help ='Last checkpoint epoch')

    return parser.parse_known_args()
if __name__ == '__main__':
    arg, unknown = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

    if arg.type == 'u':
        from Models.model_nodelevel import NodeHyperlink
        type = 'undirected'
    else:
        type = 'directed'
        from Models.model_nodelevel_directed import NodeDiHyperlink
    random.seed(arg.seed)
    Path("./nodeevent_directed/").mkdir(parents=True, exist_ok=True)

    files = ['enron', 'eumail', 'hepth', 'arxiv', 'twitter', 'bitcoin']
    path = '../../directed_datasets/'
    step = None
    if files[arg.dataset] == 'enron':
        file_name = path + 'enron/'
    elif files[arg.dataset] == 'eumail':
        file_name = path + 'eumail/'
    elif files[arg.dataset] == 'hepth':
        file_name = path + 'hepth/'
    elif files[arg.dataset] == 'arxiv':
        file_name = path + 'arxiv_25/'
        #step = 60 * 60
    elif files[arg.dataset] == 'twitter':
        file_name = path + 'twitter/'
    elif files[arg.dataset] == 'bitcoin':
        file_name = path + 'bitcoin/'
    else:
        raise "Not implemented for the rest of the dataset"
    data = HyperDiEdgeDataset(file_name, step, type=type)
    max_nodes = (data.n_nodes, data.n_nodes)

    if arg.use_wandb:
        import wandb
        wandb.init(project="Directed", entity="Submission", name= arg.fileID, group= files[arg.dataset])

    Path("./nodeevent_directed/" + files[int(arg.dataset)] + "/" + arg.fileID +"/").mkdir(parents=True, exist_ok=True)
    Path("./save_models/").mkdir(parents=True, exist_ok=True)
    save_path = "./nodeevent_directed/" + files[int(arg.dataset)] + "/"  + arg.fileID +"/"

    # Create and configure logger
    logging.basicConfig(filename=save_path + files[int(arg.dataset)] + '_' + arg.fileID + '_' +  '_' + str(arg.freq) + '_' + str(arg.seed) + '_' + type, format='%(asctime)s %(message)s', filemode='w')
    file_name =  save_path + arg.fileID + 'gcn ' + "Dataset {0}.txt mask {1} NegEdge {2}  EmbSize {3} type {4} seed {5}".format(
       files[arg.dataset], arg.mask, arg.neg_e, arg.k, type, arg.seed)
    BEST_MODEL_PATH = f'./save_models/best_model_{files[arg.dataset]}_mask{arg.mask}_NegEdge{arg.neg_e}_EmbSize{arg.k}_freq{ arg.freq}_seed{arg.seed}_{arg.model_suffix}'
    print(file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(file_name)

    p_right = data.p_right
    p_left = data.p_left
    p = [p_right, p_left]

    Neg_per_Edge = arg.neg_e

    k = arg.k
    lr = arg.lr
    num_nodes = data.n_nodes
    batch_size = 128
    batch_ids = data.batch_ids
    N_batch = len(batch_ids)
    N_train_Batches = len(batch_ids) // 2
    np.random.seed(0)
    initial_embeddings = np.random.randn(data.n_nodes + 1, k).astype('float32') * 0
    np.random.seed(arg.seed)
    device = 'cuda'
    torch.manual_seed(0)
    d_model = k
    n_head = 4
    d_k = d_model // n_head
    d_v = d_model // n_head
    arch = 'x'
    if type == 'directed':
        model = NodeDiHyperlink(n_head, k, k // n_head, k // n_head, data.n_nodes, p, data.log_inter_event_t_diff_mean, data.log_inter_event_t_diff_std, arch, arg.mask, device, factor=[arg.freq, 10], alpha = arg.alpha).to(device)
        tnbrs_right = NeighborFinder(data.all_events, data.neighbors[0])
        tnbrs_left = NeighborFinder(data.all_events, data.neighbors[1])
        model.tnbrs_right = tnbrs_right
        model.tnbrs_left = tnbrs_left
        columns = ['cur_time','right_size', 'left_size', 'MRR', 'loss', 'mae', 'mse',  'True', 'Estimate_Right', 'LastUpdate_Right', 'Conn Proba Right', 'Conn Proba Left']
    else:
        model = NodeHyperlink(n_head, k, k // n_head, k // n_head, data.n_nodes, arg.mask, device, data.p_undir, data.log_inter_event_t_diff_mean, data.log_inter_event_t_diff_std, factor=[arg.freq, 10], alpha = arg.alpha).to(device)
        tnbrs = NeighborFinder(data.all_events, data.neighbors)  # Temporal history finding
        model.tnbrs = tnbrs
        columns =  ['cur_time','right_size', 'left_size', 'MRR', 'loss', 'mae', 'mse',  'True', 'Estimate', 'LastUpdate', 'Conn Proba']
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    no_of_epochs = arg.epoch
    min_val_loss = 1e9
    best_epoch = -1
    backup_mem = None

    if arg.use_last_checkpoint:
        print('Loading Best Model')
        model.load_state_dict(torch.load(BEST_MODEL_PATH+'.pt')) 
        with open(BEST_MODEL_PATH+'_memory_data.pkl','rb') as f:
            mem_data = pickle.load(f)
        with open(BEST_MODEL_PATH+'_last_seen.pkl','rb') as f:
            last_seen = pickle.load(f)
        with open(BEST_MODEL_PATH+'_messages.pkl','rb') as f:
            messages = pickle.load(f)  
        backup_mem = (torch.FloatTensor(mem_data).to(device), torch.FloatTensor(last_seen).to(device), messages )
        model.encoder.memory.restore_memory(backup_mem)
    
    for epoch in range(arg.start_epoch + 1, no_of_epochs):
        epoch_loss_tr = 0
        epoch_hyper_loss_tr = 0
        epoch_time_loss_tr = 0
        epoch_size_loss_tr = 0
        epoch_conn_loss_tr = 0
        epoch_mrr_tr = 0
        epoch_mae_tr = 0
        epoch_scaled_mse_tr = 0
        N_train = 0
        model.train()
        iteration = tqdm(range(N_train_Batches))
        model.encoder.memory.__init_memory__(initial_embeddings)
        predictions_train = []
        auc_scores_train = []
        auc_labels_train = []
        # undirected
        size_true_train = []
        size_pred_train = []
        # directed
        size_right_true_train = []
        size_left_true_train = []
        size_right_pred_train = []
        size_left_pred_train = []

        for i in iteration:
            batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives = batching_data(batch_ids[i], data)
            loss, mrr, t_mae, t_scaled_mse, predictions_train_batch, auc_scores_batch, auc_labels_batch, \
            size_true_batch, size_pred_batch  = model.train_batch_continuous_directed(batch_pos_hyperedge, \
                                                                  batch_time_cur, batch_time_prev, batch_connectives, Neg_per_Edge, max_nodes, p, data.degree, 'train',  g_type= arg.g_type)
            
            if i > 0.1 * len(iteration) or files[arg.dataset] != 'eumail':
                N_train = N_train + len(batch_pos_hyperedge)
                optimizer.zero_grad()
                total_loss = loss[0] + loss[1] + loss[2] + loss[3]
                (total_loss/len(batch_pos_hyperedge)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss_tr = epoch_loss_tr + total_loss.item()
                epoch_hyper_loss_tr = epoch_hyper_loss_tr + loss[0].item()
                epoch_time_loss_tr = epoch_time_loss_tr + loss[1].item()
                epoch_size_loss_tr = epoch_size_loss_tr + loss[2].item()
                epoch_conn_loss_tr = epoch_conn_loss_tr + loss[3].item()
                epoch_mrr_tr = epoch_mrr_tr + mrr.item()
                epoch_mae_tr = epoch_mae_tr + t_mae.item()
                epoch_scaled_mse_tr = epoch_scaled_mse_tr + t_scaled_mse.item()
                iteration.set_postfix(loss=total_loss.item())
                predictions_train = predictions_train + predictions_train_batch
                auc_scores_train = auc_scores_train + auc_scores_batch
                auc_labels_train = auc_labels_train + auc_labels_batch
                if type == 'undirected':
                    size_true_train = size_true_train + size_true_batch
                    size_pred_train.append(size_pred_batch)
                else:
                    size_right_true_train = size_right_true_train + size_true_batch[0]
                    size_left_true_train = size_left_true_train + size_true_batch[1]
                    size_right_pred_train.append(size_pred_batch[0])
                    size_left_pred_train.append(size_pred_batch[1])

            model.encoder.memory.detach_memory()
        #scheduler.step()
        fpr, tpr, thresholds = metrics.roc_curve(auc_labels_train, auc_scores_train, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        epoch_avg_time_rmse_tr = np.sqrt(epoch_scaled_mse_tr/N_train, dtype=np.float64)
        if type == 'undirected':
            size_pred_train = np.squeeze(np.concatenate(size_pred_train, axis =0))
            size_auc_ovo_macro_tr = computeAucRocScore(size_true_train, size_pred_train, 'ovo', 'macro')
            size_auc_ovr_macro_tr = computeAucRocScore(size_true_train, size_pred_train, 'ovr', 'macro')
            size_auc_ovr_micro_tr = computeAucRocScore(size_true_train, size_pred_train, 'ovr', 'micro')
            size_auc_roc_score = str([size_auc_ovo_macro_tr, size_auc_ovr_macro_tr, size_auc_ovr_micro_tr])
        else:
            size_left_pred_train = np.squeeze(np.concatenate(size_left_pred_train, axis =0))
            size_left_auc_ovo_macro_tr = computeAucRocScore(size_left_true_train, size_left_pred_train, 'ovo', 'macro')
            size_left_auc_ovr_macro_tr = computeAucRocScore(size_left_true_train, size_left_pred_train, 'ovr', 'macro')
            size_left_auc_ovr_micro_tr = computeAucRocScore(size_left_true_train, size_left_pred_train, 'ovr', 'micro')
        
            if files[arg.dataset] != 'enron' and  files[arg.dataset] != 'eumail' and files[arg.dataset] != 'twitter':
                size_right_pred_train = np.squeeze(np.concatenate(size_right_pred_train, axis =0))
                size_right_auc_ovo_macro_tr = computeAucRocScore(size_right_true_train, size_right_pred_train, 'ovo', 'macro')
                size_right_auc_ovr_macro_tr = computeAucRocScore(size_right_true_train, size_right_pred_train, 'ovr', 'macro')
                size_right_auc_ovr_micro_tr = computeAucRocScore(size_right_true_train, size_right_pred_train, 'ovr', 'micro')
                size_auc_ovo_macro_tr = (size_right_auc_ovo_macro_tr + size_left_auc_ovo_macro_tr)/ 2
                size_auc_ovr_macro_tr = (size_right_auc_ovr_macro_tr + size_left_auc_ovr_macro_tr)/ 2
                size_auc_ovr_micro_tr = (size_right_auc_ovr_micro_tr + size_left_auc_ovr_micro_tr)/ 2
                size_auc_roc_score = str([size_auc_ovo_macro_tr, size_auc_ovr_macro_tr, size_auc_ovr_micro_tr])
            else:
                size_auc_ovo_macro_tr = size_left_auc_ovo_macro_tr
                size_auc_ovr_macro_tr = size_left_auc_ovr_macro_tr
                size_auc_ovr_micro_tr = size_left_auc_ovr_micro_tr
                size_auc_roc_score = str([size_auc_ovo_macro_tr, size_auc_ovr_macro_tr, size_auc_ovr_micro_tr])

        if arg.use_wandb:
            wandb.log({
                'Train Loss' : epoch_loss_tr / N_train,
                'Train Hyperedge Loss' : epoch_hyper_loss_tr / N_train,
                'Train Time Loss' : epoch_time_loss_tr / N_train,
                'Train Size Loss' : epoch_size_loss_tr / N_train,
                'Train Connectivity Loss' : epoch_conn_loss_tr / N_train,
                'Train MAE': epoch_mae_tr / N_train,
                'Train RMSE': epoch_avg_time_rmse_tr,
                'Train MRR': epoch_mrr_tr / N_train,
                'Train AUC':auc_score,
                'Train Size AUC (ovo macro)': size_auc_ovo_macro_tr,
                'Train Size AUC (ovr macro)': size_auc_ovr_macro_tr,
                'Train Size AUC (ovr micro)': size_auc_ovr_micro_tr,
            })
        print("Epoch  {0:d}, N_train {1:d} Loss Train = {2:.2f} Train Time Loss ={3:.3f} Train Size Loss={4:.3f}  Train Conn Loss={5:.3f} MAE Train = {6:0.4f} RMSE Train = {7:0.4f}  MRR Train = {8:0.4f} AUC Train = {9:0.4f}, Size AUC Train = {10}".format(epoch,
                                                                                                                N_train, 
                                                                                                                epoch_loss_tr / N_train,
                                                                                                                epoch_time_loss_tr / N_train,
                                                                                                                epoch_size_loss_tr / N_train,
                                                                                                                epoch_conn_loss_tr / N_train,
                                                                                                                epoch_mae_tr / N_train,
                                                                                                                epoch_avg_time_rmse_tr,
                                                                                                                epoch_mrr_tr / N_train, 
                                                                                                                auc_score, size_auc_roc_score
                                                                                                                ))
        logger.info("Epoch  {0:d}, N_train {1:d} Loss Train = {2:.2f} Train Time Loss ={3:.3f} Train Size Loss={4:.3f}  Train Conn Loss={5:.3f} MAE Train = {6:0.4f} RMSE Train = {7:0.4f}  MRR Train = {8:0.4f} AUC Train = {9:0.4f}, Size AUC Train = {10}".format(epoch,
                                                                                                                N_train, 
                                                                                                                epoch_loss_tr / N_train,
                                                                                                                epoch_time_loss_tr / N_train,
                                                                                                                epoch_size_loss_tr / N_train,
                                                                                                                epoch_conn_loss_tr / N_train,
                                                                                                                epoch_mae_tr / N_train,
                                                                                                                epoch_avg_time_rmse_tr,
                                                                                                                epoch_mrr_tr / N_train, 
                                                                                                                auc_score, size_auc_roc_score
                                                                                                                ))
        
        # predictions_train = pd.DataFrame(data=np.array(predictions_train), columns=['right_size', 'left_size', 'MRR', 'loss','mae','True', 'Estimate'])
        # predictions_train.to_csv(file_name + '_train' + '.csv', index=False)

        if epoch % 10 == 0 or epoch == 99:
            Valid_time = (N_batch + N_train_Batches) // 2
            iteration = tqdm(range(N_train_Batches, Valid_time))
            Neg_per_Edge_test = arg.neg_e
            epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te, epoch_size_loss_te, epoch_conn_loss_te, \
            epoch_mae_loss_te, epoch_scaled_mse_loss_te, epoch_mrr_loss_te,  N_test, predictions_test, \
            auc_scores_test, auc_labels_test, size_true_test, size_pred_test \
                = model.testing_continuous_directed(iteration, batch_ids, data, Neg_per_Edge_test, g_type= arg.g_type)
            
            fpr, tpr, thresholds = metrics.roc_curve(auc_labels_test, auc_scores_test, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)
            epoch_avg_time_rmse_te = np.sqrt(epoch_scaled_mse_loss_te/N_test, dtype=np.float64)
            if type == 'undirected':
                size_auc_ovo_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovo', 'macro')
                size_auc_ovr_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'macro')
                size_auc_ovr_micro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'micro')
                size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])
            else:
                size_left_auc_ovo_macro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovo', 'macro')
                size_left_auc_ovr_macro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovr', 'macro')
                size_left_auc_ovr_micro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovr', 'micro')

                if files[arg.dataset] != 'enron' and  files[arg.dataset] != 'eumail' and files[arg.dataset] != 'twitter':         
                    size_right_auc_ovo_macro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovo', 'macro')
                    size_right_auc_ovr_macro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovr', 'macro')
                    size_right_auc_ovr_micro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovr', 'micro')

                    size_auc_ovo_macro_te = (size_right_auc_ovo_macro_te + size_left_auc_ovo_macro_te)/ 2
                    size_auc_ovr_macro_te = (size_right_auc_ovr_macro_te + size_left_auc_ovr_macro_te)/ 2
                    size_auc_ovr_micro_te = (size_right_auc_ovr_micro_te + size_left_auc_ovr_micro_te)/ 2
                    size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])

                else:
                    size_auc_ovo_macro_te = size_left_auc_ovo_macro_te
                    size_auc_ovr_macro_te = size_left_auc_ovr_macro_te
                    size_auc_ovr_micro_te = size_left_auc_ovr_micro_te
                    size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])



            if arg.use_wandb:
                wandb.log({
                    'Val Loss': epoch_loss_te / N_test,
                    'Val Hyperedge Loss': epoch_hyper_loss_te / N_test,
                    'Val Time Loss': epoch_time_loss_te / N_test,
                    'Val Size Loss': epoch_size_loss_te / N_test,
                    'Val Conn Loss': epoch_conn_loss_te / N_test,
                    'Val MAE': epoch_mae_loss_te / N_test, 
                    'VAL RMSE': epoch_avg_time_rmse_te,
                    'Val MRR': epoch_mrr_loss_te / N_test, 
                    'Val AUC':auc_score,
                    'Val Size AUC (ovo macro)': size_auc_ovo_macro_te,
                    'Val Size AUC (ovr macro)': size_auc_ovr_macro_te,
                    'Val Size AUC (ovr micro)': size_auc_ovr_micro_te,

                })
            print( "Epoch  {0:d} N_Valid {1:d} Loss Valid = {2:.2f} Val Time Loss ={3:.3f} Val Size Loss ={4:.3f} Val Conn Loss={5:.3f} MAE Valid = {6:.4f} RMSE Valid = {7:.4f} MRR Valid = {8:.4f} AUC Valid ={9:.5f} Size AUC Valid ={10} ".format(epoch,
                                                                                                                    N_test, \
                                                                                                                    epoch_loss_te / N_test, \
                                                                                                                    epoch_time_loss_te / N_test,
                                                                                                                    epoch_size_loss_te / N_test,
                                                                                                                    epoch_conn_loss_te / N_test,
                                                                                                                    epoch_mae_loss_te / N_test, 
                                                                                                                    epoch_avg_time_rmse_te,
                                                                                                                    epoch_mrr_loss_te / N_test, 
                                                                                                                    auc_score, size_auc_roc_score_te,
                                                                                                                    ))
            logger.info( "Epoch  {0:d} N_Valid {1:d} Loss Valid = {2:.2f} Val Time Loss ={3:.3f} Val Size Loss ={4:.3f} Val Conn Loss={5:.3f} MAE Valid = {6:.4f} RMSE Valid = {7:.4f} MRR Valid = {8:.4f} AUC Valid ={9:.5f} Size AUC Valid ={10} ".format(epoch,
                                                                                                                    N_test, \
                                                                                                                    epoch_loss_te / N_test, \
                                                                                                                    epoch_time_loss_te / N_test,
                                                                                                                    epoch_size_loss_te / N_test,
                                                                                                                    epoch_conn_loss_te / N_test,
                                                                                                                    epoch_mae_loss_te / N_test, 
                                                                                                                    epoch_avg_time_rmse_te,
                                                                                                                    epoch_mrr_loss_te / N_test, 
                                                                                                                    auc_score, size_auc_roc_score_te
                                                                                                                    ))
            # if (epoch % 10 == 0 or epoch ==99) and files[arg.dataset] != 'eumail' and  files[arg.dataset] != 'bitcoin':
            #     # predictions_train = pd.DataFrame(data=np.array(predictions_train), columns=columns)
            #     # predictions_train.to_csv(file_name + f'_train{epoch}' + '.csv', index=False)
            #     # predictions_valid = pd.DataFrame(data=np.array(predictions_test),
            #     #                                  columns=columns)
            #     # predictions_valid.to_csv(file_name + f'_valid{epoch}' + '.csv', index=False)
            #     save_predictions(file_name + f'_train{epoch}' + '.txt', predictions_train, columns)
            #     save_predictions(file_name + f'_valid{epoch}' + '.txt', predictions_test, columns)

            if min_val_loss > epoch_loss_te / N_test:
                torch.save(model.state_dict(),BEST_MODEL_PATH+'.pt')
                backup_mem = model.encoder.memory.backup_memory()
                with open(BEST_MODEL_PATH+'_memory_data.pkl','wb') as f:
                    pickle.dump(backup_mem[0].detach().cpu().tolist(), f)
                with open(BEST_MODEL_PATH+'_last_seen.pkl','wb') as f:
                    pickle.dump(backup_mem[1].detach().cpu().tolist(), f)
                with open(BEST_MODEL_PATH+'_messages.pkl','wb') as f:
                    pickle.dump(backup_mem[2], f)

                best_epoch = epoch
                # predictions_train = pd.DataFrame(data=np.array(predictions_train), columns= columns)
                # predictions_train.to_csv(file_name + f'_train' + '.csv', index=False)
                # predictions_valid = pd.DataFrame(data=np.array(predictions_test),
                #                                  columns= columns)
                # predictions_valid.to_csv(file_name + f'_valid' + '.csv', index=False)
                # save_predictions(file_name + f'_train' + '.txt', predictions_train, columns)
                # save_predictions(file_name + f'_valid' + '.txt', predictions_test, columns)
                min_val_loss = epoch_loss_te / N_test
                logger.info(f'Saving the best mode at {best_epoch} with val loss {min_val_loss}')
                print(f'Saving the best mode at {best_epoch} with val loss {min_val_loss}')

    model.load_state_dict(torch.load(BEST_MODEL_PATH+'.pt')) 
    # with open(BEST_MODEL_PATH+'_memory_data.pkl','rb') as f:
    #     mem_data = pickle.load(f)
    # with open(BEST_MODEL_PATH+'_last_seen.pkl','rb') as f:
    #     last_seen = pickle.load(f)
    # with open(BEST_MODEL_PATH+'_messages.pkl','rb') as f:
    #     messages = pickle.load(f)  
    # backup_mem = (torch.FloatTensor(mem_data).to(device), torch.FloatTensor(last_seen).to(device), messages )
    model.encoder.memory.restore_memory(backup_mem)
    iteration = tqdm(range(Valid_time, N_batch))

    epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te, epoch_size_loss_te, epoch_conn_loss_te,\
    epoch_mae_loss_te, epoch_scaled_mse_loss_te,  epoch_mrr_loss_te, N_test, predictions_test, \
    auc_scores_test, auc_labels_test,size_true_test, size_pred_test \
        = model.testing_continuous_directed(iteration, batch_ids, data, Neg_per_Edge_test, g_type= arg.g_type)
    fpr, tpr, thresholds = metrics.roc_curve(auc_labels_test, auc_scores_test, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    epoch_avg_time_rmse_te = np.sqrt(epoch_scaled_mse_loss_te/N_test, dtype=np.float64)
    if type == 'undirected':
        size_auc_ovo_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovo', 'macro')
        size_auc_ovr_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'macro')
        size_auc_ovr_micro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'micro')
        size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])
    else:
        size_left_auc_ovo_macro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovo', 'macro')
        size_left_auc_ovr_macro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovr', 'macro')
        size_left_auc_ovr_micro_te = computeAucRocScore(size_true_test[1], size_pred_test[1], 'ovr', 'micro')

        if files[arg.dataset] != 'enron' and  files[arg.dataset] != 'eumail' and files[arg.dataset] != 'twitter':          
            size_right_auc_ovo_macro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovo', 'macro')
            size_right_auc_ovr_macro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovr', 'macro')
            size_right_auc_ovr_micro_te = computeAucRocScore(size_true_test[0], size_pred_test[0], 'ovr', 'micro')

            size_auc_ovo_macro_te = (size_right_auc_ovo_macro_te + size_left_auc_ovo_macro_te)/ 2
            size_auc_ovr_macro_te = (size_right_auc_ovr_macro_te + size_left_auc_ovr_macro_te)/ 2
            size_auc_ovr_micro_te = (size_right_auc_ovr_micro_te + size_left_auc_ovr_micro_te)/ 2
            size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])

        else:
            size_auc_ovo_macro_te = size_left_auc_ovo_macro_te
            size_auc_ovr_macro_te = size_left_auc_ovr_macro_te
            size_auc_ovr_micro_te = size_left_auc_ovr_micro_te
            size_auc_roc_score_te = str([size_auc_ovo_macro_te, size_auc_ovr_macro_te, size_auc_ovr_micro_te])

    if arg.use_wandb:
        wandb.log({
            'Test Loss': epoch_loss_te / N_test,
            'Test Hyperedge Loss': epoch_hyper_loss_te / N_test,
            'Test Time Loss': epoch_time_loss_te / N_test,
            'Test Size Loss': epoch_size_loss_te / N_test,
            'Test Conn Loss': epoch_conn_loss_te / N_test,
            'Test MSE': epoch_mae_loss_te / N_test, 
            'Test RMSE': epoch_avg_time_rmse_te, 
            'Test MRR': epoch_mrr_loss_te / N_test, 
            'Test AUC':auc_score,
            'Test Size AUC (ovo macro)': size_auc_ovo_macro_te,
            'Test Size AUC (ovr macro)': size_auc_ovr_macro_te,
            'Test Size AUC (ovr micro)': size_auc_ovr_micro_te
        })
    
    print( "Best Epoch  {0:d} N_Test {1:d} Loss Test = {2:.2f} Test Time Loss ={3:.3f} Test Size Loss ={4:.3f} Test Conn Loss={5:.3f} MAE Test = {6:.2f} RMSE Test ={7:.4f} MRR Test = {8:.4f} AUC Test ={9:.5f} Size AUC Test={10}".format(best_epoch,
                                                                                                            N_test, \
                                                                                                            epoch_loss_te / N_test, \
                                                                                                            epoch_time_loss_te / N_test,
                                                                                                            epoch_size_loss_te / N_test,
                                                                                                            epoch_conn_loss_te / N_test,
                                                                                                            epoch_mae_loss_te / N_test, 
                                                                                                            epoch_avg_time_rmse_te,
                                                                                                            epoch_mrr_loss_te / N_test, 
                                                                                                            auc_score, size_auc_roc_score_te,
                                                                                                            ))
    logger.info( "Best Epoch  {0:d} N_Test {1:d} Loss Test = {2:.2f} Test Time Loss ={3:.3f} Test Size Loss ={4:.3f} Test Conn Loss={5:.3f} MAE Test = {6:.2f} RMSE Test ={7:.4f} MRR Test = {8:.4f} AUC Test ={9:.5f} Size AUC Test={10}".format(best_epoch,
                                                                                                            N_test, \
                                                                                                            epoch_loss_te / N_test, \
                                                                                                            epoch_time_loss_te / N_test,
                                                                                                            epoch_size_loss_te / N_test,
                                                                                                            epoch_conn_loss_te / N_test,
                                                                                                            epoch_mae_loss_te / N_test, 
                                                                                                            epoch_avg_time_rmse_te,
                                                                                                            epoch_mrr_loss_te / N_test, 
                                                                                                            auc_score, size_auc_roc_score_te,
                                                                                                            ))
        
    # predictions_test = pd.DataFrame(data=np.array(predictions_test),
    #                                     columns= columns)
    # predictions_test.to_csv(file_name + '_test' + '.csv', index=False)
    save_predictions(file_name + f'_test' + '.txt', predictions_test, columns)
