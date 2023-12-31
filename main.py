
from dataloader import generate_dataloaders
from model import *
import trainer
import trainer_plus
import torch
import os
import numpy as np
import argparse
from parameters import Parameter

#setting seed
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


#Input parameters
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type = str, required=True)
parser.add_argument("--dataset", type=str, default="SleepEEG")
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--model", type = str, default="TFC")

args, unknown = parser.parse_known_args()


training_mode = args.mode
dataset = args.dataset
debug = args.debug
model = args.model



data_path = f"datasets/{dataset}"
experiment_log_dir = "saved_models/" + training_mode + f"_seed_{SEED}"
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)



parameters = {
    "SleepEEG": Parameter(128, 60, 5, 178, "SleepEEG"),
    "Epilepsy": Parameter(128, 60, 2, 178, "Epilepsy"),
    "Gesture": Parameter(128, 60, 8, 178, "Gesture")
    }



train_dl, valid_dl, test_dl = generate_dataloaders(data_path, 
                                                   training_mode, 
                                                   parameters[dataset].batch_size, 
                                                   parameters[dataset].target_batch_size, 
                                                   debug)

TFC_model = TFC(
    parameters[dataset].EncoderParams, 
    parameters[dataset].projectorParams,
    )


classifier = target_classifier(parameters[dataset].num_classes)

TFC_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)

#experiment log dir is left
if training_mode == "pretrain":
    trainer.PreTrainer(TFC_model, TFC_optimizer, train_dl, experiment_log_dir)
elif training_mode == "fine_tune":
    load_from = f"saved_models/pretrain_seed_{SEED}/ckp_last.pt"
    print("The loading file path", load_from)
    chkpoint = torch.load(load_from)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)
    if model == 'TFC':
        trainer.FineTuner(TFC_model, TFC_optimizer, valid_dl, classifier, classifier_optimizer, test_dl, "SleepEEG2"+parameters[dataset].name)
    elif model == 'TAMPo':
        trainer_plus.FineTuner(TFC_model, TFC_optimizer, valid_dl, classifier, classifier_optimizer, test_dl, "SleepEEG2"+parameters[dataset].name)
    else:
        print("model invalid, try again")
else:
    print("invalid mode, try again")