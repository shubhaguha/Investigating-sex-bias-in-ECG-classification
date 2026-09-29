# Imports
import argparse
import sys
import json
import pickle
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from data.data_loader import *
from models.resnet_attention import *
from models.cnn import *
from fastai.vision.models.xresnet import xresnet101
from utils.train import *
from utils.focal_loss import *
import numpy as np



# Arguments to be passed from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='Provide necessary arguments for training the model')

    parser.add_argument('-d', '--data_dir', type=str, default='data/PhysioNet2021_preprocessed',
                        help="Directory containing the preprocessed WFDB_* database folders")
    parser.add_argument('-dd', '--division_file', type=str, default=None,
                        help="JSON file with the train/val/test split per fold (default: <data_dir>/dataset_division.json)")
    parser.add_argument('-r', '--sex_ratio', type=str, default='50_50',
                        choices=['100_0', '75_25', '50_50', '25_75', '0_100'],
                        help="The sex procentage ratio in the training data males vs females, options: [100_0, 75_25, 50_50, 25_75, 0_100]")
    parser.add_argument('-f', '--fold', type=int, default=0,
                        help="fold to be use as a test set(0-4)")
    parser.add_argument('-m', '--model', type=str, default='xresnet101',
                        choices=['cnn', 'resnet_attention', 'xresnet101'],
                        help="Classifier algorithm, choose from [cnn, resnet_attention, xresnet101]")
    parser.add_argument('-exp', '--experiment_id', type=str, default='test',
                        help="Experiment ID")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help="Number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('-loss', '--loss', type=str, default='focal', choices=['bce', 'focal'],
                        help="Loss function: ['bce', 'focal']")
    parser.add_argument('-p', '--patience', type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument('-n', '--normalize', type=str, default='z-score',
                        help="Normalize the data: ['z-score', 'min_max', None]")
    parser.add_argument('-t', '--length', type=int, default=4096,
                        help="Length of the signal")
    parser.add_argument('-c', '--class_nr', type=int, default=3,
                        help="Number of classes")
    parser.add_argument('-g', '--gpu', type=int, nargs='+', default=0,
                        help="GPU number(s)")
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('-alpha', '--alpha', type=float, default=0.75,
                        help="alpha for focal loss")
    parser.add_argument('-beta', '--beta', type=int, default=2,
                        help="beta for focal loss")

    return vars(parser.parse_args())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_model(args):
    if args["model"] == 'resnet_attention':
        model = ResnetAttention(args["class_nr"])
    elif args["model"] == 'xresnet101':
        model = xresnet101(pretrained=False, c_in=12, ndim=1, n_out=args["class_nr"])
    elif args["model"] == 'cnn':
        model = CNN(num_labels=args["class_nr"], length=args["length"])
    else:
        print('Model not defined')
        sys.exit()

    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=1e-5)
    if args["loss"] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args["loss"] == 'focal':
        criterion = FocalLoss(alpha = args["alpha"], gamma=args["beta"])
        print(criterion)
    return model, optimizer, criterion

def save_args(args, results_dir):
    with open(f'{results_dir}/args.json', 'w') as f:
        json.dump(args, f, indent=4)

def setup_directories(experiment_id):
    results_dir = f'results/{experiment_id}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def main():
    args = parse_args()
    print(args)
    set_seed(args["seed"])
    results_dir = setup_directories(args["experiment_id"])
    save_args(args, results_dir)

    data_directory = [os.path.join(args["data_dir"], db) for db in
                      ['WFDB_PTBXL', 'WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_ChapmanShaoxing', 'WFDB_Ningbo']]

    print('Finding header and recording files...')
    print(data_directory)
    header_files, recording_files = find_challenge_files(data_directory)
    cinc_dataset = dataset(header_files, nr_leads=12, length=args["length"], normalize=args["normalize"], equivalent_cl='sinus_mi', return_source=False)
    classes_info(cinc_dataset)

    # getting the dataset division
    # Path to the JSON file with dataset division for various traing ratios and folds
    file_path = args["division_file"] or os.path.join(args["data_dir"], "dataset_division.json")

    with open(file_path, "r") as f:
        dataset_division = json.load(f)[str(args['fold'])]
        print(args['fold'])
    male_test_idx = dataset_division["male_balanced_test_idx"]
    female_test_idx = dataset_division["female_balanced_test_idx"]

    train_idx = dataset_division[f"train_idx_{args['sex_ratio']}"]
    val_idx = dataset_division[f"val_idx_{args['sex_ratio']}"]

    train = Subset(cinc_dataset, train_idx)
    val = Subset(cinc_dataset, val_idx)
    male_test = Subset(cinc_dataset, male_test_idx)
    female_test = Subset(cinc_dataset, female_test_idx)


    train_loader = DataLoader(dataset=train, batch_size=args["batch_size"], shuffle=True, collate_fn=collate,
                              num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(dataset=val, batch_size=args["batch_size"], shuffle=True, collate_fn=collate,
                            num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
    male_test_loader = DataLoader(dataset=male_test, batch_size=args["batch_size"], shuffle=False, collate_fn=collate,
                             num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
    female_test_loader = DataLoader(dataset=female_test, batch_size=args["batch_size"], shuffle=False, collate_fn=collate,
                                  num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)

    model, optimizer, criterion = setup_model(args)

    #multiple gpus if needed
    gpus = [args['gpu']] if type(args['gpu']) == int else args['gpu']
    DEVICE = torch.device(f"cuda:{gpus[0]}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available() and len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    print(args)
    print(DEVICE)

    train_loop(model, train_loader, val_loader, args["epochs"], args["patience"], optimizer, criterion,
               DEVICE, args["class_nr"], args["experiment_id"])
    # Female test set
    print('Female test set')
    female_results = test_loop(model, female_test_loader, DEVICE, args["class_nr"])
    # Male test set
    print('Male test set')
    male_results = test_loop(model, male_test_loader, DEVICE, args["class_nr"])

    with open(os.path.join(results_dir, 'test_results.pickle'), 'wb') as f:
        pickle.dump({'female': female_results, 'male': male_results}, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()