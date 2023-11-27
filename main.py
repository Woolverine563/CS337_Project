
from dataloader import generate_dataloaders
from model import *
from trainer import *
data_path = "datasets/SleepEEG/"
mode = "pre_training"
batch_size = 128 # For Sleep EEG, batch_size = 128

train_dl, valid_dl, test_dl = generate_dataloaders(data_path, mode, batch_size = batch_size)

encoder = EncoderParams(178)
projector = ProjectorParams(178)
output_dim = 2
TFC_model = TFC(encoder, projector)
classifier = target_classifier(output_dim)

training_mode = "pretrain"
TFC_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)

if training_mode == "pretrain":
    PreTrainer(TFC_model, TFC_optimizer, train_dl)
else:
    load_from = os.path.join(os.path.join('experiments_logs/finetunemodel/sleepedf2eplipsy_model.pt'))
    print("The loading file path", load_from)
    chkpoint = torch.load(load_from)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)
    FineTuner(TFC_model, TFC_optimizer,valid_dl, classifier, classifier_optimizer, test_dl)
