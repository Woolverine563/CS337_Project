
from dataloader import generate_dataloaders
from model import *
from trainer import *
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

training_mode = parser.mode
dataset = parser.dataset
debug = parser.debug



data_path = f"datasets/{dataset}"
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_2layertransformer")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)



parameters = {
    "SleepEEG": Parameter(128, 60, 2, 178)
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
classifier = target_classifier(parameters[dataset].output_dim)

TFC_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)

#experiment log dir is left
if training_mode == "pretrain":
    PreTrainer(TFC_model, TFC_optimizer, train_dl, experiment_log_dir)
elif training_mode == "fine_tune":
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}_2layertransformer", "saved_models"))
    print("The loading file path", load_from)
    chkpoint = torch.load(load_from)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)
    FineTuner(TFC_model, TFC_optimizer,valid_dl, classifier, classifier_optimizer, test_dl)
else:
    print("invalid mode, try again")