import torch 
from torch import nn
from loss import ContrastiveLoss
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import sys
sys.path.append("..")

def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    if(training_mode == "pre_train"):
        PreTrainer(model, model_optimizer, train_dl,experiment_log_dir)
    else:
        FineTuner(model, model_optimizer, valid_dl, classifier, classifier_optimizer, test_dl)

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def PreTrainer(model, model_optimizer, train_dl,experiment_log_dir, num_epochs = 40, tau = 0.2, lam = 1/6):

    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        model_optimizer.zero_grad()
        # print(epoch)
        # print(train_dl)
        for batch_num, (data_t, labels, aug_t, data_f, aug_f) in enumerate(train_dl):
            # print("batch:", batch_num)
            data_t, aug_t = data_t.float(), aug_t.float()
            data_f, aug_f = data_f.float(), aug_f.float()

            #embeddings:
            h_t, z_t, h_f, z_f = model(data_t, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug_t, aug_f)
            
            #initilizing loss class
            loss_function = ContrastiveLoss(data_t.shape[0], tau)
            #calc_loss
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

    chkpoint = {'model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, 'ckp_last.pt'))
    print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'/ckp_last.pt'))

def supervised_contrastive_loss(similarities, labels, tau):
    """
    TODO: Implement the supervised contrastive loss
    Inputs:
    similarities: shape: (2N, 2N), where N = batch size 
    labels: shape: (2N,)
    tau: scalar (hyperparameter) set to 0.1

    Returns:
    loss: scalar value, unnormalized supervised contrastive loss computed across the batch
    """
    sims = torch.exp(similarities / tau)
    n = sims.shape[0]
    denoms = torch.sum(sims, axis = 1) - sims[range(n), range(n)]
    l = -torch.log(sims / denoms.unsqueeze(1))
    l_sum = torch.zeros(1)
    for i in range(n):
        # if(i % 2 == 0):
        #     l_sum += l[i][i + 1]
        # else:
        #     l_sum += l[i][i - 1]
        set_size = 0
        term = torch.zeros(1)
        for j in range(n):
            if i == j:
                continue
            if labels[i] == labels[j]:
                set_size += 1
                term += l[i][j]
        l_sum += term / set_size    

    return l_sum / n

cosine = nn.CosineSimilarity(dim=-1)

def get_sims(vec1, vec2):

    size = 2*vec1.shape[0]
    dims = vec1.shape[1]

    data_2n = torch.zeros(size, dims)  # 512 is the model output embedding size
    # labels_2n = torch.zeros(size, dtype=torch.long, device=device)

    """
    TODO: Save the embeddings of the image and its transform for the current batch into data_2n and labels_2n
    Note: augmentation passed as an argument to contrastive_training is used to create a transform of the input 
    """
    
    data_2n[range(0, size, 2)] = vec1
    data_2n[range(1, size, 2)] = vec2

    # for i in range(size // 2):
    #     labels_2n[2 * i] = labels[i]
    #     labels_2n[2 * i + 1] = labels[i]

    similarities = cosine(data_2n.unsqueeze(0), data_2n.unsqueeze(1))
    return similarities
            
def FineTuner(model,model_optimizer, val_dl, classifier, classifier_optimizer, test_dl, arch, num_epochs = 40, tau = 0.2, lam = 1/21, mu = 10/21):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    f1_scores = []
    for epoch in range(num_epochs):

        model_optimizer.param_groups[0]['lr'] = 1e-4
        classifier_optimizer.param_groups[0]['lr'] = 1e-4

        valid_loss, data_finetune, labels_finetune, F1 = finetune(model, model_optimizer, classifier, 
                                                                  classifier_optimizer, val_dl, tau, lam, mu)

        #storing best model
        scheduler.step(valid_loss)
        if len(f1_scores) == 0 or F1 > max(f1_scores):
            print('Updating fine-tuned model!')
            os.makedirs('saved_models/fine_tune_seed_42/', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/fine_tune_seed_42/' + arch + '_model.pt')
            torch.save(classifier.state_dict(), 'saved_models/fine_tune_seed_42/' + arch + '_classifier.pt')
        f1_scores.append(F1)
        #not loading model always
        #not running knn always
        #printing diff test

    #evaluation on test set
    model.load_state_dict(torch.load('saved_models/fine_tune_seed_42/' + arch + '_model.pt'))
    classifier.load_state_dict(torch.load('saved_models/fine_tune_seed_42/' + arch + '_classifier.pt'))
    data_test, labels_test, performance = model_test(model,classifier, test_dl)    
    
    print('---------------------------------------------------------\n')
    print('Best Testing Performance: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f '
              '| AUPRC=%.4f' % (performance[0], performance[1], performance[2], performance[3],
                                performance[4], performance[5]))
    print('\n---------------------------------------------------------')
    
    """Use KNN as another classifier; it's an alternation of the MLP classifier in function model_test. 
    Experiments show KNN and MLP may work differently in different settings, so here we provide both. """
    # train classifier: KNN
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(data_finetune, labels_finetune)
    # # print('KNN finetune acc:', knn_acc_train)
    # knn_test = data_test.detach().cpu().numpy()

    # knn_result = neigh.predict(knn_test)
    # knn_result_score = neigh.predict_proba(knn_test)
    # one_hot_label_test = one_hot_encoding(labels_test)
    # # print(classification_report(label_test, knn_result, digits=4))
    # # print(confusion_matrix(label_test, knn_result))
    # knn_acc = accuracy_score(labels_test, knn_result)
    # precision = precision_score(labels_test, knn_result, average='macro', )
    # recall = recall_score(labels_test, knn_result, average='macro', )
    # F1 = f1_score(labels_test, knn_result, average='macro')
    # auc = roc_auc_score(one_hot_label_test, knn_result_score, average="macro", multi_class="ovr")
    # prc = average_precision_score(one_hot_label_test, knn_result_score, average="macro")
    # print('KNN Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'%
    #         (knn_acc, precision, recall, F1, auc, prc))

def finetune(model, model_optimizer, classifier, classifier_optimizer, val_dl, tau, lam, mu):
        #model_finetune
        model.train()
        classifier.train()
        CE_loss = nn.CrossEntropyLoss()
    
        accuracy_score = []
        auc_scores = []
        prc_scores = []
        epoch_losses = []

        pred_arr = np.array([])
        labels_numpy = np.array([])
        data_arr = np.array([])
        #might need to see this, no enumerate
        #issue

        for data_t, labels, aug_t, data_f, aug_f in val_dl:
        #issue over
            #data format
            data_t, aug_t = data_t.float(), aug_t.float()
            data_f, aug_f = data_f.float(), aug_f.float()
            labels = labels.long()
            # print(labels)
            #optimizers
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            #embeddings:
            h_t, z_t, h_f, z_f = model(data_t, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug_t, aug_f)
            
            #initilizing loss class
            loss_function = ContrastiveLoss(data_t.shape[0], tau)

            #calc_loss
            # loss_time = loss_function(h_t, h_t_aug)
            # loss_frq = loss_function(h_f, h_f_aug)
            # loss_tfconsistency = loss_function(z_t, z_f)

            """Now start with classifier"""
            
            size = 2*z_t.shape[0]

            labels_2n = torch.zeros(size, dtype=torch.long)

            for i in range(size // 2):
                labels_2n[2 * i] = labels[i]
                labels_2n[2 * i + 1] = labels[i]

            similarities = get_sims(h_t_aug, h_t)
            loss_ht_htaug = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_ht_htaug = torch.div(loss_ht_htaug, size)

            similarities = get_sims(h_f_aug, h_f)
            loss_hf_hfaug = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_hf_hfaug = torch.div(loss_hf_hfaug, size)

            similarities = get_sims(z_t, z_f)
            loss_zt_zf = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_zt_zf = torch.div(loss_zt_zf, size)

            similarities = get_sims(h_t, h_t_aug)
            loss_ht_htaug = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_ht_htaug = torch.div(loss_ht_htaug, size)

            similarities = get_sims(h_f, h_f_aug)
            loss_hf_hfaug = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_hf_hfaug = torch.div(loss_hf_hfaug, size)

            similarities = get_sims(z_t_aug, z_f_aug)
            loss_ztaug_zfaug = supervised_contrastive_loss(similarities, labels_2n, 0.2)
            loss_ztaug_zfaug = torch.div(loss_ztaug_zfaug, size)

            #format input to classifier
            #flattend directly
            #issue
            input = torch.cat((z_t, z_f), dim = 1)
            #issue over
            # print(input.shape)

            #predictions
            preds = classifier(input)
            input = input.reshape(input.shape[0], -1)
            #calculating final loss
            # print(preds.shape)
            # print(labels.shape)
            loss_p = CE_loss(preds, labels)
            # loss = (1-mu-lam)*loss_p+ mu*loss_tfconsistency + lam*(loss_time + loss_frq)
            # print("reached")
            loss_time = loss_function(h_t, h_t_aug)
            loss_frq = loss_function(h_f, h_f_aug)
            # loss = loss_zt_zf + loss_ztaug_zfaug + 2*loss_p + loss_ht_htaug + loss_hf_hfaug
            print(loss_p)
            print(loss_zt_zf, loss_ztaug_zfaug)
            loss = loss_zt_zf + loss_ztaug_zfaug + loss_p
            # loss = loss_p
            accuracy = labels.eq(preds.detach().argmax(dim=1)).float().mean()
            onehot_label = nn.functional.one_hot(labels)
            pred_numpy = preds.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
            except:
                auc_bs = np.float(0)
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)

            accuracy_score.append(accuracy)
            auc_scores.append(auc_bs)
            prc_scores.append(prc_bs)
            epoch_losses.append(loss.item())

            loss.backward()
            model_optimizer.step()
            classifier_optimizer.step()

            pred_arr = np.append(pred_arr, pred_numpy)
            labels_numpy = labels.data.cpu().numpy()
            data_arr = np.append(data_arr, input.data.cpu().numpy())
        
        data_arr = data_arr.reshape([len(labels_numpy), -1])  # produce the learned embeddings
        pred_numpy = np.argmax(pred_numpy, axis=1)
        precision = precision_score(labels_numpy, pred_numpy, average='macro', )
        recall = recall_score(labels_numpy, pred_numpy, average='macro', )
        F1 = f1_score(labels_numpy, pred_numpy, average='macro', )
        ave_loss = torch.tensor(epoch_losses).mean()
        ave_acc = torch.tensor(accuracy_score).mean()
        ave_auc = torch.tensor(auc_scores).mean()
        ave_prc = torch.tensor(prc_scores).mean()
        print('Finetune: loss = %.4f| Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | AUPRC = %.4f'
            % (ave_loss, ave_acc*100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_prc *100))

        return ave_loss,data_arr, labels_numpy, F1

def model_test(model, classifier, test_dl):
    #evaluate model on test_dl
    model.eval()
    classifier.eval()

    #auc and prc
    total_auc = []
    total_prc = []
    data_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data_t, labels, _,data_f, _ in test_dl:
            data_t, labels = data_t.float(), labels.long()
            data_f = data_f.float()

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data_t, data_f)
            input = torch.cat((z_t, z_f), dim = 1)
            predictions_test = classifier(input)
            input = input.reshape(input.shape[0], -1)
            data_all.append(input)

            onehot_label = nn.functional.one_hot(labels)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,average="macro", multi_class="ovr")
            except:
                auc = np.float(0)
            prc = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_auc.append(auc)
            total_prc.append(prc)

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    # print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'
    #       % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))
    data_all = torch.concat(tuple(data_all))
    return data_all, labels_numpy_all, performance
            
