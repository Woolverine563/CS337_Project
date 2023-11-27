import torch 
from loss import ContrastiveLoss


def PreTrainer(model, model_optimizer, train_dl, num_epochs = 40, tau = 0.2, lam = 0.25):

    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        model_optimizer.zero_grad()
        
        for batch_num, (data_t, labels, aug_t, data_f, aug_f) in enumerate(train_dl):
            data_t, aug_t = data_t.float(), aug_t.float()
            data_f, aug_f = data_f.float(), aug_f.float()

            #embeddings:
            h_t, z_t, h_f, z_f = model(data_t, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug_t, aug_f)
            
            #initilizing loss class
            loss_function = ContrastiveLoss(data_t.shape[0], tau)

            loss_time = loss_function(h_t, h_t_aug)
            loss_frq = loss_function(h_f, h_f_aug)
            loss_tfconsistency = loss_function(z_t, z_f)

            #improvable, can add z_t-z_f_aug\ z_t_aug-z_f\ z_f_aug-z_t_aug\
            loss = lam*(loss_time + loss_frq) + (1-lam)*loss_tfconsistency
            #improvable over
            epoch_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
        
        #printing
        print(f"Epoch: {epoch}, loss = {torch.tensor(epoch_loss).mean().item()}")
            

