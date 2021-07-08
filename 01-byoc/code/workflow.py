import os
import time
import copy
import datetime
import json
from argparse import ArgumentParser
from pprint import pformat
import logging

import shutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import plot_confusion_matrix
from dataset import SoundDataset
from dataloader import SoundDataLoader
from config import ParameterSetting
from models import VGGish, EfficientNet_model
from metrics import accuracy, f1, roc_auc, cfm, classification_report
from losses import CrossEntropyLoss
from ops import Adam, StepLR
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

import pkbar

logger = logging.getLogger(__file__)


def get_optim_scheduler(params, model):
    # optimizer
    if params.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=params.lr)
    # scheduler
    if params.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=int(params.epochs*0.8), gamma=0.1)
    return optimizer, scheduler


def get_folder_name(params):
    # description of model and folder name
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d-%H_%M")
    model_name = "{0:}_lr-{1:.0e}_optim-{2:}_scheduler-{3:}".format(
                folder_name, params.lr,
                params.optimizer, params.scheduler)
    save_model_path = os.path.join(params.save_root, "snapshots", model_name)
    return save_model_path, model_name


def train_model(model, params, dataloaders, dataset_sizes, all_params):
    ####################
    # training setting #
    ####################

    optimizer, scheduler = get_optim_scheduler(params, model)
    save_model_path, model_name = get_folder_name(params)

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
        print("create folder: {}".format(save_model_path))

    log_path = os.path.join(params.save_root, "log", model_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Now using device = ', device)
    # print('params = ', all_params)

    since = time.time()

    best_f1 = 0.0
    best_auc = 0.0
    best_true, best_pred, best_prob = [], [], []



    print('Calculating mean and std')

    i = 0

    for data in tqdm(dataloaders['train']):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs_stack = inputs.cpu() if i == 0 else torch.vstack((inputs_stack, inputs.cpu()))
        i += 1


    mean_train, std_train = torch.mean(inputs_stack, dim=0), torch.std(inputs_stack, dim=0)

    np.savez(os.path.join(save_model_path, 'mean_std.npz'), mean=mean_train, std=std_train)

    print('Tensorboard log:  ' + 'tensorboard --logdir=/' + str(log_path) + '/--port 6099')

    
    

    ####################
    #  start training  #
    ####################

    last_model_path = None 
    for epoch in range(params.epochs):
        print('Epoch {}/{}'.format(epoch+1, params.epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # set model to train/eval model
            model.train() if phase == 'train' else model.eval()
            # set progress bar
            kbar = pkbar.Kbar(target=(dataset_sizes[phase]//params.batch_size)+1, width=8)

            running_loss = 0.0
            # prediction and groundtruth label
            y_true, y_pred, y_prob = [], [], []
            start_time = time.time()
            # iterative training
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = (inputs - mean_train.to(device)) / std_train.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # compute loss
                    loss = CrossEntropyLoss(outputs, labels)
                    # backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        Binarize_Y_test = label_binarize(labels.cpu().numpy(), classes=list(range(params.num_class)))
                        n_classes = Binarize_Y_test.shape[1]
                        if batch_idx == 0:
                            stacked_outputs = outputs.cpu().numpy()
                            stacked_Binarize_Y_test = Binarize_Y_test
                        else:
                            stacked_outputs = np.vstack((stacked_outputs, outputs.cpu().numpy()))
                            stacked_Binarize_Y_test = np.vstack((stacked_Binarize_Y_test, Binarize_Y_test))

                gt_label_in_batch = labels.data.cpu().detach().numpy()
                running_loss += loss.item() * inputs.size(0)

                y_true.extend(gt_label_in_batch)
                y_pred.extend(preds.cpu().detach().numpy())
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                y_prob.extend(outputs.cpu().detach().numpy())

                if phase == 'train':
                    kbar.update(batch_idx, values=[("train loss in batch", loss)])
                    writer.add_scalar('train loss', loss, epoch*len(dataloaders[phase]) + batch_idx)
                else:
                    kbar.update(batch_idx, values=[("val loss in batch", loss)])
                    writer.add_scalar('val loss', loss, epoch*len(dataloaders[phase]) + batch_idx)
                
            
            # finish an epoch
            time_elapsed = time.time() - start_time
            print()
            print("finish this epoch in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            # compute classification results in an epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy(y_true, y_pred)
            epoch_f1 = f1(y_true, y_pred)
#            epoch_roc_auc = roc_auc(y_true, y_prob) # remove to avoid error occurrs in sparse input 

            if phase == 'train':
                scheduler.step()
                kbar.add(1, values=[("train epoch loss", epoch_loss), ("train acc", epoch_acc), ("train f1", epoch_f1)])
                writer.add_scalar('train accuracy', epoch_acc, epoch)
                writer.add_scalar('train f1 score', epoch_f1, epoch)
            else:
                kbar.add(1, values=[("val epoch loss", epoch_loss), ("val acc", epoch_acc), ("val f1", epoch_f1)])
                writer.add_scalar('val accuracy', epoch_acc, epoch)
                writer.add_scalar('val f1 score', epoch_f1, epoch)

                # save model if f1 and precision are all the best
                if epoch_f1 > best_f1 :
                    best_f1 = epoch_f1 if epoch_f1 > best_f1 else best_f1
                    best_true = y_true
                    best_pred = y_pred
                    best_prob = y_prob
                    wpath = os.path.join(save_model_path, 'epoch_{:03d}_valloss_{:.4f}_valacc_{:.4f}_f1_{:.4f}.pkl'.format(epoch+1, epoch_loss, epoch_acc, epoch_f1))
                    #torch.save(model.state_dict(), wpath)
                    torch.save(model, wpath)
                    last_model_path = wpath
                    print("=== save weight " + wpath + " ===")
                print()

                ######################################AUC Calculate############################
                from sklearn.metrics import roc_auc_score
                from sklearn.metrics import roc_curve, auc
                # Compute ROC curve and ROC area for each class

                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for class_i in range(n_classes):
                    fpr[class_i], tpr[class_i], _ = roc_curve(stacked_Binarize_Y_test[:, class_i], stacked_outputs[:, class_i])
                    roc_auc[class_i] = auc(fpr[class_i], tpr[class_i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(stacked_Binarize_Y_test.ravel(), stacked_outputs.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                ######################################AUC Calculate############################

                if epoch == 0:
                    roc_auc_all = roc_auc
                    
                else:
                    for roc_auc_i in roc_auc.keys():
                        temp_list = [roc_auc_all[roc_auc_i]]
                        temp_list.append(roc_auc[roc_auc_i])
                        roc_auc_all[roc_auc_i] = temp_list

                for print_i in list(range(params.num_class)):
                    print('class ' + str(print_i) + ': ', str(roc_auc[print_i]))

                print('Average:', roc_auc['micro'])

                if roc_auc['micro'] > best_auc :
                    best_auc = roc_auc['micro']
                
                writer.add_scalar('val Auc score', roc_auc['micro'], epoch)
                    
    

    model_file = params.model_file
    shutil.move(last_model_path, model_file)
    ##############
    # evaluation #
    ##############

    # finish training
    
    target_names = ["Barking", "Howling", "Crying", "COSmoke","GlassBreaking","Other"]
    time_elapsed = time.time() - since
    cfmatrix = cfm(best_true, best_pred)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(accuracy(best_true, best_pred)))
    print('Best val F1: {:4f}'.format(f1(best_true, best_pred)))
    print('Best val Auc: {:4f}'.format(best_auc))

    print(cfmatrix)

    with open(os.path.join(log_path, "classification_report.txt"), "w") as f:
        f.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+"\n")
        f.write('Best val Acc: {:4f}'.format(accuracy(best_true, best_pred))+"\n")
        f.write('Best val F1: {:4f}'.format(f1(best_true, best_pred))+"\n")
        f.write('Best val Auc: {:4f}'.format(best_auc)+"\n")
        f.write(str(cfmatrix)+"\n")
        f.write('All Parameters :' +  str(all_params))

    plot_confusion_matrix(cfmatrix, target_names, log_path)

def prepare_model(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(params)
    print("build model...")
    model = None
    if params.resume:
        model = torch.load(params.resume)
    elif params.model_name == 'VGGish':
        model = VGGish(params)
    elif params.model_name == 'Efficientnet':
        model = EfficientNet_model(params).return_model()

    model = model.to(device)
    return model