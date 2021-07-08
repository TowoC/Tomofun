import os
import time
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pprint import pformat
import pandas as pd
import logging

import torch
from torch.utils.data import DataLoader

from dataset import SoundDataset
from config import ParameterSetting
from models import VGGish, EfficientNet_model
from metrics import cfm, classification_report, roc_auc

logger = logging.getLogger(__file__)


def main():
    parser = ArgumentParser()
    # arguments for test
    parser.add_argument("--test_csv", type=str, default='/home/tingwei/Tomofun/sample_submission.csv')
    parser.add_argument("--data_dir", type=str, default='/home/tingwei/Tomofun/public_test', help="the directory of sound data")
    parser.add_argument("--aug_dir", type=str, default ='/home/tingwei/Tomofun/urbansound')
    # parser.add_argument("--np_dir", type=str, default='/home/tingwei/Tomofun/Final/incremental-training-mlops_rebuild/01-byoc/code/results/snapshots/2021-07-06-13_46_lr-5e-04_optim-adam_scheduler-steplr/mean_std.npz', help='the mean and std numpy file')
    parser.add_argument("--model_name", type=str, default='Efficientnet', choices=['VGGish', 'Efficientnet'], help='the algorithm we used')
    parser.add_argument("--model_path", nargs="+", default=['/home/tingwei/Tomofun/Final/incremental-training-mlops_rebuild/01-byoc/code/results/snapshots/2021-07-06-13_46_lr-5e-04_optim-adam_scheduler-steplr/epoch_025_valloss_0.2963_valacc_0.9250_f1_0.9194.pkl'])
    parser.add_argument("--batch_size", type=int, default=128, help="the batch size")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num_class", type=int, default=6, help="number of classes")
    parser.add_argument("--saved_root", type=str, default='results/test', help="the path of test results.")
    parser.add_argument("--saved_name", type=str, default='test_results', help="the prefix of test files")
    # proprocessing setting
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--nfft", type=int, default=200)
    parser.add_argument("--hop", type=int, default=80)
    parser.add_argument("--mel", type=int, default=64)
    parser.add_argument("--normalize", type=str, default=None, choices=[None, 'rms', 'peak'], help="normalize the input before fed into model")
    parser.add_argument("--preload", action='store_true', default=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    params = ParameterSetting(csv_path=args.test_csv, data_dir=args.data_dir, aug_dir=args.aug_dir, batch_size=args.batch_size, num_class=args.num_class, sr=args.sr,
                              nfft=args.nfft, hop=args.hop, mel=args.mel, normalize=args.normalize, preload=args.preload)

    ###################
    # model preparing #
    ###################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.model_name == 'VGGish':
        model = VGGish(params)
    elif args.model_name == 'Efficientnet':
        model = EfficientNet_model(params).return_model()

    ##################
    # data preparing #
    ##################
    test_df = pd.read_csv(params.csv_path)

    print("Preparing testing data...")

    dataset = SoundDataset(params, test_df, proba = 0, label_flag = False)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

    print("the number of wavfiles : {}".format(len(dataset)))

    


    ##################
    #  test the file #
    ##################

    # print(args.model_path)

    for model_idx, model_name in enumerate(args.model_path):
        # model.load_state_dict(torch.load(model_name))
        model = torch.load(model_name)
        m_s = np.load(os.path.join(os.path.split(model_name)[0], 'mean_std.npz'))
        mean_train = torch.tensor(m_s['mean'])
        std_train = torch.tensor(m_s['std'])


        model.eval()
        model = model.to(device)

        y_pred, y_true, y_prob = [], [], []
        with torch.no_grad():
            since = time.time()
            for batch_idx, data in tqdm(enumerate(dataloader)):
                data = data.to(device)
                data = (data - mean_train.to(device)) / data.to(device)
                outputs = model(data)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                pred_label = preds.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                # gt = gt.data.cpu().detach().numpy()

                # y_true.extend(gt)
                y_pred.extend(pred_label)
                y_prob.extend(outputs)

            time_elapsed = time.time() - since
            print('test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(y_prob[:5])
        # print(cfm(y_true, y_pred))
        # print(classification_report(y_true, y_pred))
        # print(roc_auc(y_true, y_prob))


        if not os.path.exists(args.saved_root):
            os.mkdir(args.saved_root)

        # with open(os.path.join(args.saved_root, "{}_{}.txt".format(args.saved_name, model_idx)), 'w') as f:
        #     f.write(str(cfm(y_true, y_pred))+"\n")
        #     f.write(classification_report(y_true, y_pred)+"\n")
        #     f.write("roc auc score: "+str(roc_auc(y_true, y_prob))+"\n")

if __name__ == '__main__':
    main()
