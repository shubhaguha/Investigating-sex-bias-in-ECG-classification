#from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pickle
import os
import numpy as np
import torch
import copy
import torch.optim as optim


def needs_lead_indicator(model):
    # Unwrap nn.DataParallel so the check also works on multi-GPU runs
    model = getattr(model, 'module', model)
    return 'ResnetAttention' in model.__class__.__name__


def train_loop(model, train_loader, val_loader, num_epochs, patience, optimizer, criterion, DEVICE, class_nr, experiment_ID = None):
    OUTPUT = []

    # Variables for early stopping
    best_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

    model_directory = 'results/model_weights/'
    os.makedirs(model_directory, exist_ok=True)

    writer = SummaryWriter(comment=str('_' + experiment_ID))
    # Ensure the model is moved to the DEVICE
    model.to(DEVICE)
    previous_lr = optimizer.param_groups[0]['lr']
    print('First lr:', previous_lr)
    for epoch in range(num_epochs):
        model.train()
        roc_epoch = [0] * class_nr
        temp_loss = 0
        for local_batch, local_labels, lead_idx in train_loader:
            local_labels = local_labels.float().to(DEVICE)
            lead_idx = lead_idx.float().to(DEVICE)
            # Run the forward pass
            if needs_lead_indicator(model):
                local_batch = local_batch.unsqueeze(2).float().to(DEVICE)
                outputs = model(local_batch, lead_idx).float()
            else:
                local_batch = local_batch.float().to(DEVICE)
                outputs = model(local_batch).float()

            loss = criterion(outputs, local_labels)
            temp_loss += loss.cpu().detach().numpy()

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ROC AUC train
            temp_roc = []
            for i in range(class_nr):
                if len(np.unique(local_labels.cpu().detach().numpy()[:, i])) > 1:
                    temp_roc.append(
                        roc_auc_score(local_labels.cpu().detach().numpy()[:, i], outputs.cpu().detach().numpy()[:, i]))
                else:
                    temp_roc.append(1)
            roc_epoch = [sum(x) for x in zip(roc_epoch, temp_roc)]
            del local_batch, local_labels, outputs, temp_roc

        train_roc_auc = [round(x / len(train_loader), 2) for x in roc_epoch]
        train_loss = round(temp_loss / len(train_loader), 4)

        # Validation
        roc_epoch_valid = [0] * class_nr
        val_temp_loss = 0

        for val_local_batch, val_local_labels, lead_idx in val_loader:
            val_local_labels = val_local_labels.float().to(DEVICE)
            lead_idx = lead_idx.float().to(DEVICE)
            # Run the forward
            model.eval()
            with torch.no_grad():
                if needs_lead_indicator(model):
                    val_local_batch = val_local_batch.unsqueeze(2).float().to(DEVICE)
                    val_outputs = model(val_local_batch, lead_idx).float()
                else:
                    val_local_batch = val_local_batch.float().to(DEVICE)
                    val_outputs = model(val_local_batch).float()
                val_loss = criterion(val_outputs, val_local_labels.float())
                val_temp_loss += val_loss.cpu().detach().numpy()
                val_temp_roc = []
                for i in range(class_nr):
                    if len(np.unique(val_local_labels.cpu().detach().numpy()[:, i])) > 1:
                        val_temp_roc.append(roc_auc_score(val_local_labels.cpu().detach().numpy()[:, i],
                                                          val_outputs.cpu().detach().numpy()[:, i]))
                    else:
                        val_temp_roc.append(1)
                roc_epoch_valid = [sum(x) for x in zip(roc_epoch_valid, val_temp_roc)]
            del val_local_batch, val_local_labels, val_outputs, val_temp_roc
            #torch.cuda.empty_cache()

        val_roc_auc = [round(x / len(val_loader), 2) for x in roc_epoch_valid]
        val_loss = round(val_temp_loss / len(val_loader), 4)
        #writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)

        print("----- Epoch ", epoch, "-----")
        print("Training loss: ", train_loss)
        print("Train ROC AUC: ", train_roc_auc)
        print("Val loss: ", val_loss)
        print("Valid ROC AUC: ", val_roc_auc)

        OUTPUT.append({'epoch': epoch,
                       'train_rocauc': train_roc_auc,
                       'train_loss': train_loss,
                       'val_rocauc': val_roc_auc,
                       'val_loss': val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0 # reset trigger times if loss improved
        else:
            trigger_times += 1

        rounded_loss = round(val_loss, 3)
        scheduler.step(rounded_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")
        if current_lr != previous_lr:
            print(f"Learning rate decreased from {previous_lr} to {current_lr} !")
            model.load_state_dict(best_model_wts)
            trigger_times = 0
        previous_lr = current_lr

        # Early stopping
        if optimizer.param_groups[0]['lr'] < 0.0001: #trigger_times >= patience and
            print('Early stopping at epoch {}'.format(epoch + 1))
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), model_directory + experiment_ID + '.pth')
            break

        # Reaching the epoch limits without early stopping
        if epoch == num_epochs-1:
            print('Last epoch')
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), model_directory + experiment_ID + '.pth')
            break


    writer.flush()
    writer.close()

    path_name = Path(model_directory, f'PROGRESS_{experiment_ID}.pickle')

    with open(path_name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def test_loop(model, test_loader, DEVICE, class_nr):
    TEST = []
    test_roc = []
    p_roc_auc = []
    pr_auc = []
    model.eval()
    labels = []
    outputs = []
    for test_local_batch, test_local_labels, lead_idx in test_loader:
        test_local_labels = test_local_labels.float().to(DEVICE)
        lead_idx = lead_idx.float().to(DEVICE)
        # Run the forward pass
        with torch.no_grad():
            if needs_lead_indicator(model):
                test_local_batch = test_local_batch.unsqueeze(2).float().to(DEVICE)
                test_outputs = model(test_local_batch, lead_idx).float()
            else:
                test_local_batch = test_local_batch.float().to(DEVICE)
                test_outputs = model(test_local_batch).float()

        #with torch.no_grad():
            labels.append(test_local_labels.cpu().detach().numpy().tolist())
            outputs.append(test_outputs.cpu().detach().numpy().tolist())
        del test_local_batch, test_local_labels

    labels = np.array([val for sublist in labels for val in sublist])
    outputs = np.array([val for sublist in outputs for val in sublist])

    for i in range(class_nr):
        if len(np.unique(labels[:, i])) > 1:
            test_roc.append(roc_auc_score(labels[:, i], outputs[:, i]))
            p_roc_auc.append(roc_auc_score(labels[:, i], outputs[:, i], max_fpr=0.2))
            pr_auc.append(average_precision_score(labels[:, i], outputs[:, i]))
        else:
            test_roc.append(1)
            p_roc_auc.append(1)
            pr_auc.append(1)

    TEST.append({'labels': labels,
                   'outputs': outputs,
                   'test_roc': test_roc,
                   'p_roc_auc': p_roc_auc,
                   'pr_auc': pr_auc})

    print("Results on test set:")
    print([round(x, 2) for x in test_roc])
    print([round(x, 2) for x in p_roc_auc])
    print([round(x, 2) for x in pr_auc])
    return TEST[0]



