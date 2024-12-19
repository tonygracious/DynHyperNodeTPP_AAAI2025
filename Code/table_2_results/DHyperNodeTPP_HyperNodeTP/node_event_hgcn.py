import argparse
import os
import logging
from tqdm import tqdm

from DataLoader.Dataloader import HyperEdgeDataset
from Utils.utils import batching_data, computeAucRocScore
from Utils.history import NeighborFinder
import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path

from sklearn import metrics


def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='node event')
    parser.add_argument('--fileID', type=str, default='node event')
    parser.add_argument('--dataset', type=int, default='0')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--mask', default=True, action='store_true')
    parser.add_argument('--neg_e', type=int, default=20)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freq', type=float, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--loss', type=str, default='likeli')
    parser.add_argument('--epoch',type=int, default= 100)
    parser.add_argument("--use_wandb" , action="store_true")
    parser.add_argument('--alpha', type=float, default= 0.3, help='Alpha Hyperparameter In Connectivity Model')
    parser.add_argument("--model_suffix", type=str , default="JohnDoe", help ='Suffix to add to model before saving it')
    return parser.parse_known_args()


if __name__ == '__main__':
    arg, unknown = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    print(arg)
    fileIDs = ['enron', 'contact', 'threads', 'congress', 'eumail', 'ndc', 'ndcsub']
    if arg.use_wandb:
        import wandb 
        wandb.init(project="Undirected", entity="Submission", name= arg.fileID, group= fileIDs[arg.dataset])
    
    from Models.model_nodelevel import NodeHyperlink

    Path("./Results/").mkdir(parents=True, exist_ok=True)
    Path("./save_models/").mkdir(parents=True, exist_ok=True)
    random.seed(arg.seed)
    # Create and configure logger
    Path("./Results/" + fileIDs[int(arg.dataset)] + "/" + arg.fileID + "/").mkdir(parents=True, exist_ok=True)
    save_path = "./Results/" + fileIDs[int(arg.dataset)] + "/" + arg.fileID + "/"
    logging.basicConfig(
        filename=save_path + fileIDs[int(arg.dataset)] + '_' + arg.fileID + '_' + arg.loss + '_' + str(arg.freq) + '_' + str(
            arg.seed), format='%(asctime)s %(message)s', filemode='w')
    file_name = save_path + arg.fileID + '_' + arg.loss + \
                "GCN Conti Dataset {0}.txt mask {1} NegEdge {2} EmbSize {3} freq {4} batch {5}  seed {6}".format(
                    fileIDs[arg.dataset], arg.mask, arg.neg_e, arg.k, arg.freq, arg.batch, arg.seed)
    print(file_name)
    lr = arg.lr
    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    logger.info(file_name)
    path = "../DynHyperGraph/"
    BEST_MODEL_PATH = f'./save_models/best_model_{fileIDs[arg.dataset]}_mask{arg.mask}_NegEdge{arg.neg_e}_EmbSize{arg.k}_freq{ arg.freq}_batch{arg.batch}_seed{arg.seed}_{arg.model_suffix}.pt'
    file_names_size_file = ["email-Enron/email-Enron-nverts.txt", "contact-high-school/contact-high-school-nverts.txt", \
                            "threads-math-sx/threads-math-sx-nverts.txt", "congress-bills/congress-bills-nverts.txt", \
                            "email-Eu/email-Eu-nverts.txt", "NDC-classes/NDC-classes-nverts.txt",
                            "NDC-substances/NDC-substances-nverts.txt"]
    file_names_simplex_file = ["email-Enron/email-Enron-simplices.txt",
                               "contact-high-school/contact-high-school-simplices.txt", \
                               "threads-math-sx/threads-math-sx-simplices.txt",
                               "congress-bills/congress-bills-simplices.txt", \
                               "email-Eu/email-Eu-simplices.txt", "NDC-classes/NDC-classes-simplices.txt",
                               "NDC-substances/NDC-substances-simplices.txt"]
    file_names_times_file = ["email-Enron/email-Enron-times.txt", "contact-high-school/contact-high-school-times.txt", \
                             "threads-math-sx/threads-math-sx-times.txt", "congress-bills/congress-bills-times.txt", \
                             "email-Eu/email-Eu-times.txt", "NDC-classes/NDC-classes-times.txt",
                             "NDC-substances/NDC-substances-times.txt"]
    file_names_labels = ["email-Enron/email-Enron-node-labels.txt", None,
                         "threads-math-sx/threads-math-sx-simplex-labels.txt", None, None, None, None]

    file_Id = int(arg.dataset)
    simplex_size_file = path + file_names_size_file[file_Id]
    simplex_file = path + file_names_simplex_file[file_Id]
    times_file = path + file_names_times_file[file_Id]

    if file_Id in [1, 3, 4, 5, 6]:
        labels_file = None
        step = 1
        if file_Id == 5:
            step = 86400000.0
        if file_Id == 4:
            step = None
        if file_Id == 6:
            step = 86400000.0
    else:
        step = None
        labels_file = path + file_names_labels[file_Id]

    data = HyperEdgeDataset(simplex_size_file, simplex_file, times_file, labels_file, step=step, normalize_time=True)
    batch_size = arg.batch
    batch_ids = data.batch_ids
    N_batch = len(batch_ids)

    Neg_per_Edge = arg.neg_e
    k = arg.k
    N_train_Batches = len(batch_ids) // 2
    device = 'cuda'
    np.random.seed(0)
    initial_embeddings = np.random.randn(data.n_nodes + 1, k).astype('float32') * 0 # Use nn.Embedding inside encoder instead of this
    torch.manual_seed(0)
    n_heads = 4
    np.random.seed(arg.seed)
    model = NodeHyperlink(n_heads, k, k // n_heads, k // n_heads, data.n_nodes, arg.mask, device, data.p,
                          data.log_inter_event_t_diff_mean, data.log_inter_event_t_diff_std, factor=[arg.freq, 10], alpha = arg.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    no_of_epochs = arg.epoch
    tnbrs = NeighborFinder(data.all_events, data.neighbors)#Temporal history finding
    model.tnbrs = tnbrs
    min_val_loss = 1e9
    best_epoch = -1
    for epoch in range(no_of_epochs):
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
        size_true_train = []
        size_pred_train = []
        for i in iteration:
            batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives = batching_data(batch_ids[i], data)
            loss, mrr, t_mae, t_scaled_mse, predictions_train_batch, auc_scores_batch, auc_labels_batch, size_true_batch, size_pred_batch = model.train_batch_continuous(batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives, Neg_per_Edge, data.p, mode='train', neg_sampling_mode='main')
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
            epoch_scaled_mse_tr = epoch_scaled_mse_tr + t_scaled_mse.item()
            epoch_mrr_tr = epoch_mrr_tr + mrr.item()
            epoch_mae_tr = epoch_mae_tr + t_mae.item()
            iteration.set_postfix(loss=total_loss.item())
            predictions_train = predictions_train + predictions_train_batch
            auc_scores_train = auc_scores_train + auc_scores_batch
            auc_labels_train = auc_labels_train + auc_labels_batch
            size_true_train = size_true_train + size_true_batch
            size_pred_train.append(size_pred_batch)
            model.encoder.memory.detach_memory()
        #scheduler.step()
        fpr, tpr, thresholds = metrics.roc_curve(auc_labels_train, auc_scores_train, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        epoch_avg_time_rmse_tr = np.sqrt(epoch_scaled_mse_tr/N_train, dtype=np.float64)

        size_pred_train = np.squeeze(np.concatenate(size_pred_train, axis =0))
        
        size_auc_ovo_macro_tr = computeAucRocScore(size_true_train,size_pred_train, 'ovo', 'macro')
        size_auc_ovr_macro_tr = computeAucRocScore(size_true_train,size_pred_train, 'ovr', 'macro')
        size_auc_ovr_micro_tr = computeAucRocScore(size_true_train,size_pred_train, 'ovr', 'micro')
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
        # predictions_train = pd.DataFrame(data=np.array(predictions_train),
        #                                  columns=['size', 'MRR', 'loss','mae', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
        # predictions_train.to_csv(file_name + '_train.csv', index=False)
        # if epoch % 10 == 0 or epoch == 99:
        Valid_time = (N_batch + N_train_Batches) // 2
        
        iteration = tqdm(range(N_train_Batches, Valid_time))

        epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te, epoch_size_loss_te, epoch_conn_loss_te, epoch_mae_loss_te,\
        epoch_scaled_mse_loss_te, epoch_mrr_loss_te,  N_test, predictions_test, auc_scores_test, auc_labels_test,size_true_test, \
        size_pred_test = model.testing_continuous(iteration, batch_ids, data, Neg_per_Edge)

        fpr, tpr, thresholds = metrics.roc_curve(auc_labels_test, auc_scores_test, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        epoch_avg_time_rmse_te = np.sqrt(epoch_scaled_mse_loss_te/N_test, dtype=np.float64)
        size_auc_ovo_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovo', 'macro')
        size_auc_ovr_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'macro')
        size_auc_ovr_micro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'micro')
        
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
        print( "Epoch  {0:d} N_Valid {1:d} Loss Valid = {2:.2f} Val Time Loss ={3:.3f} Val Size Loss ={4:.3f} Val Conn Loss={5:.3f} MAE Valid = {6:.4f} RMSE Valid = {7:.4f} MRR Valid = {8:.4f} AUC Valid ={9:.5f} Size AUC Valid ={10}".format(epoch,
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
        logger.info( "Epoch  {0:d} N_Valid {1:d} Loss Valid = {2:.2f} Val Time Loss ={3:.3f} Val Size Loss ={4:.3f} Val Conn Loss={5:.3f} MAE Valid = {6:.4f} RMSE Valid = {7:.4f} MRR Valid = {8:.4f} AUC Valid ={9:.5f} Size AUC Valid ={10}".format(epoch,
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
        
        # predictions_valid = pd.DataFrame(data=np.array(predictions_test),
        #                                     columns=['size', 'MRR', 'loss','mae','mse', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
        # predictions_valid.to_csv(file_name + '_valid.csv', index=False)

        if epoch % 10 == 0 or epoch ==99:
            predictions_train = pd.DataFrame(data=np.array(predictions_train),
                                        columns=['CurTime','size', 'MRR', 'loss','mae','mse', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
            predictions_train.to_csv(file_name + f'_train_{epoch}.csv', index=False)
            predictions_valid = pd.DataFrame(data=np.array(predictions_test),
                                            columns=['CurTime', 'size', 'MRR', 'loss','mae','mse', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
            predictions_valid.to_csv(file_name + f'_valid_{epoch}.csv', index=False) 

        if min_val_loss > epoch_loss_te / N_test:
            torch.save(model.state_dict(),BEST_MODEL_PATH)
            backup_mem = model.encoder.memory.backup_memory()
            best_epoch = epoch
            predictions_train = pd.DataFrame(data=np.array(predictions_train),
                                        columns=['CurTime','size', 'MRR', 'loss','mae','mse', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
            predictions_train.to_csv(file_name + '_train.csv', index=False)
            predictions_valid = pd.DataFrame(data=np.array(predictions_test),
                                            columns=['CurTime', 'size', 'MRR', 'loss','mae','mse', 'LastUpdate', 'True', 'Estimate','Conn Proba'])
            predictions_valid.to_csv(file_name + '_valid.csv', index=False) 
            min_val_loss = epoch_loss_te / N_test
            logger.info(f'Saving the best mode at {best_epoch} with val loss {min_val_loss}')
            print(f'Saving the best mode at {best_epoch} with val loss {min_val_loss}')
        
    model.load_state_dict(torch.load(BEST_MODEL_PATH))    
    model.encoder.memory.restore_memory(backup_mem)
    iteration = tqdm(range(Valid_time, N_batch))    
    epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te, epoch_size_loss_te, epoch_conn_loss_te, epoch_mae_loss_te,\
    epoch_scaled_mse_loss_te, epoch_mrr_loss_te,  N_test, predictions_test, auc_scores_test, auc_labels_test,size_true_test, \
    size_pred_test = model.testing_continuous(iteration, batch_ids, data, Neg_per_Edge)
    fpr, tpr, thresholds = metrics.roc_curve(auc_labels_test, auc_scores_test, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    epoch_avg_time_rmse_te = np.sqrt(epoch_scaled_mse_loss_te/N_test, dtype=np.float64)
    size_auc_ovo_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovo', 'macro')
    size_auc_ovr_macro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'macro')
    size_auc_ovr_micro_te = computeAucRocScore(size_true_test,size_pred_test, 'ovr', 'micro')
    
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
            'Test Size AUC (ovr micro)': size_auc_ovr_micro_te,
        })
    
    print( "Best Epoch  {0:d} N_Test {1:d} Loss Test = {2:.2f} Test Time Loss ={3:.3f} Test Size Loss ={4:.3f} Test Conn Loss={5:.3f} MAE Test = {6:.2f} RMSE Test ={7:.4f} MRR Test = {8:.4f} AUC Test ={9:.5f} Size AUC Test={10} ".format(best_epoch,
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
    logger.info( "Best Epoch  {0:d} N_Test {1:d} Loss Test = {2:.2f} Test Time Loss ={3:.3f} Test Size Loss ={4:.3f} Test Conn Loss={5:.3f} MAE Test = {6:.2f} RMSE Test ={7:.4f} MRR Test = {8:.4f} AUC Test ={9:.5f} Size AUC Test={10}".format(best_epoch,
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
        

    predictions_test = pd.DataFrame(data=np.array(predictions_test),
                                        columns=['CurTime', 'size', 'MRR', 'loss', 'mae','mse','LastUpdate','True', 'Estimate','Conn Proba'])
    predictions_test.to_csv(file_name + '_test.csv', index=False)