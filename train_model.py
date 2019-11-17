import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from tqdm import tqdm
tqdm.pandas()


def train_model(index, model, train_dataset, valid_dataset, batchsize, device):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=4)

    no_of_epochs = 8

    valid_loss_min = np.Inf
    patience = 4
    # current number of epochs, where validation loss didn't increase
    p = 0
    # whether training should be stopped
    stop = False

    since = time.time()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_losses = []
    valid_losses = []
    for epoch in tqdm(range(no_of_epochs)):
        print('Epoch {}/{}'.format(epoch, no_of_epochs - 1))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        for x_batch, y_batch in tk0:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward Pass
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item()

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        model.eval()
        valid_preds = np.zeros((len(valid_dataset), 9))
        valid_loss = 0.0
        best_val_loss = np.inf
        tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
        for i, (x_batch, y_batch) in enumerate(tk1):
            with torch.no_grad():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward Pass
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
            valid_loss += loss.item()
            valid_preds[i * valid_loader.batch_size:(i + 1) * valid_loader.batch_size, :] = preds.detach().cpu().numpy()
        epoch_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(epoch_valid_loss)

        if epoch_valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.10f} --> {:.10f}).  Saving model ... to model{}.pt'.format(
                valid_loss_min,
                epoch_valid_loss,
                index))
            torch.save(model.state_dict(), f'./model/model{index}.pt')
            valid_loss_min = epoch_valid_loss
            p = 0

        # check if validation loss didn't improve
        if epoch_valid_loss > valid_loss_min:
            p += 1
            print(f'{p} epochs of increasing val loss')
            if p > patience:
                print('Stopping training')
                stop = True
                break

        if stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'save model => model{index}.bin')
    torch.save(model.state_dict(), f'./model/model{index}.bin')
    return valid_preds, training_losses, valid_losses
