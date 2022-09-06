import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.cuda import amp
import gc
import numpy as np
import copy


def train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, device):
    model.train()
    scaler = amp.GradScaler()
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
    for i, data in bar:
        optimizer.zero_grad()
        input_ids = data["input_ids"].to(device).squeeze(1)
        attention_mask = data["attention_mask"].to(device).squeeze(1)
        token_type_ids = data["token_type_ids"].to(device).squeeze(1)
        targets = data["label"].to(device).unsqueeze(1)
        batch_size = input_ids.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(y_pred, targets.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        bar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def val_one_epoch(model, criterion, val_dataloader, device, epoch):
    model.eval()
    dataset_size = 0
    running_loss = 0.0

    val_scores = []
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation")
    for i, data in bar:
        input_ids = data["input_ids"].to(device).squeeze(1)
        attention_mask = data["attention_mask"].to(device).squeeze(1)
        token_type_ids = data["token_type_ids"].to(device).squeeze(1)
        targets = data["label"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        y_pred = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(y_pred, targets.float())
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)

        y_pred = y_pred.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        print(y_pred)
        print('----------------')
        print(targets)

        val_acc = accuracy_score(targets, y_pred)
        val_f1 = f1_score(targets, y_pred)

        val_scores.append([val_acc, val_f1])

    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


def run_training(model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, device, epochs=5):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -np.inf
    best_acc = -np.inf
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, device)
        val_loss, val_scores = val_one_epoch(model, criterion, val_dataloader, device, epoch)
        valid_acc, valid_f1 = val_scores

        print(
            f' - train_loss: {train_loss:0.4f} - val_loss: {val_loss:0.4f} - val_acc: {valid_acc:0.4f} - val_f1: {valid_f1:0.4f}')

        if valid_acc >= best_acc:
            print(f'Validation accuracy increased ({best_acc:0.4f} --> {valid_acc:0.4f}).  Saving model ...')
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"last_epoch-{best_epoch:02d}.bin"
            torch.save(model.state_dict(), PATH)

            print()
            print()

    print(f'Best val Acc: {best_acc:0.4f}, Best val F1: {best_f1:0.4f}')
    model.load_state_dict(best_model_wts)
    return model
