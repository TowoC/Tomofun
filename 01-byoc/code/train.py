import os
import time
import copy
import datetime
from pprint import pformat
import logging
import torch
import numpy as np
from dataset import SoundDataset
from dataloader import SoundDataLoader
from config import ParameterSetting
from workflow import * 
import pkbar
import pandas as pd
from sklearn.utils import shuffle

logger = logging.getLogger(__file__)


def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--csv_path", type=str, default='/home/tingwei/Tomofun/Final/meta_train.csv', help='the path of train csv file')
    parser.add_argument("--data_dir", type=str, default='/home/tingwei/Tomofun/Final/train', help="the directory of sound data")
    parser.add_argument("--aug_dir", type=str, default ='/home/tingwei/Tomofun/urbansound')
    parser.add_argument("--np_dir", type=str, default="./results/mean_std.npz", help='the mean and std numpy file')
    parser.add_argument("--save_root", type=str, default="./results", help="the root of results")
    parser.add_argument("--model_file", type=str, default="./results/final_mode.pkl", help="the root of results")
    parser.add_argument("--resume", type=str, default=None, help="the path of resume training model")
    # training parameter setting
    parser.add_argument("--model_name", type=str, default='Efficientnet', choices=['VGGish', 'Efficientnet'], help='the algorithm we used')
    parser.add_argument("--val_split", type=float, default=0.2, help="the ratio of validation set. 0 means there's no validation dataset")
    parser.add_argument("--epochs", type=int, default=20, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam"])
    parser.add_argument("--scheduler", type=str, default="steplr", choices=["steplr"])
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--num_class", type=int, default=10, help="number of classes")
    parser.add_argument("--normalize", type=str, default=None, choices=[None, 'rms', 'peak'], help="normalize the input before fed into model")
    parser.add_argument("--preload", action='store_true', default=False, help="whether to convert to melspectrogram first before start training")
    # data augmentation setting
    parser.add_argument("--spec_aug", action='store_true', default=False)
    parser.add_argument("--time_drop_width", type=int, default=64)
    parser.add_argument("--time_stripes_num", type=int, default=2)
    parser.add_argument("--freq_drop_width", type=int, default=8)
    parser.add_argument("--freq_stripes_num", type=int, default=2)
    # proprocessing setting
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--nfft", type=int, default=200)
    parser.add_argument("--hop", type=int, default=80)
    parser.add_argument("--mel", type=int, default=64)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    all_params = args

    ##################
    # config setting #
    ##################

    params = ParameterSetting(args.csv_path, args.data_dir, args.aug_dir, args.np_dir, args.save_root, args.model_file, args.model_name, args.val_split,
                              args.epochs, args.batch_size, args.lr, args.num_class,
                              args.time_drop_width, args.time_stripes_num, args.freq_drop_width, args.freq_stripes_num,
                              args.sr, args.nfft, args.hop, args.mel, args.p, args.alpha, args.resume, args.normalize, args.preload,
                              args.spec_aug, args.optimizer, args.scheduler)

    if not os.path.exists(params.save_root):
        os.mkdir(params.save_root)
        print("create folder: {}".format(params.save_root))
        if not os.path.exists(os.path.join(params.save_root, 'snapshots')):
            os.mkdir(os.path.join(params.save_root, 'snapshots'))
        if not os.path.exists(os.path.join(params.save_root, 'log')):
            os.mkdir(os.path.join(params.save_root, 'log'))

    ###################
    # model preparing #
    ###################

    model = prepare_model(params)

    ##################
    # data preparing #
    ##################
    df = pd.read_csv(params.csv_path)

    num_items = len(df)
    num_train = round(num_items * (1 - params.val_split))
    num_val = num_items - num_train

    random_df = shuffle(df, random_state = 0)

    train_df, val_df = random_df[:num_train].reset_index(drop=True), random_df[num_train:].reset_index(drop=True)

    print('Training Data Distribution')
    for i in range(6):
        print('Label ' + str(i) + ' = ' + str(len(np.where(train_df['Label']==i)[0])))

    print("Preparing training/validation data...")
    print('Training p = ' + str(params.p) + ',Validation p = '  + str(0))
    dataset_train = SoundDataset(params, train_df, proba = params.p)
    dataset_val = SoundDataset(params, val_df, proba = 0)

    train_dataloader = SoundDataLoader(dataset_train, batch_size=params.batch_size, shuffle=True, validation_split=0, num_workers = 12, pin_memory=True)
    val_dataloader = SoundDataLoader(dataset_val, batch_size=params.batch_size, shuffle=True, validation_split=0, num_workers = 4, pin_memory=True)

    # val_dataloader = train_dataloader.split_validation()

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataloader.sampler), 'val': len(val_dataloader.sampler)}
    print("train size: {}, val size: {}".format(dataset_sizes['train'], dataset_sizes['val']))

    ##################
    # model training #
    ##################

    # start to train the model
    train_model(model, params, dataloaders, dataset_sizes, all_params)

if __name__ == '__main__':
    main()
