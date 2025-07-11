import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Tuple
import numpy as np

from tqdm import tqdm
import os
import shutil
from typing import Tuple
from torch import Tensor
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adjust_learning_rate(learning_rate, optimizer, epoch, change_epoch1=100, change_epoch2=105):
    lr = learning_rate
    if epoch >= change_epoch1:
        lr /= 10
    if epoch >= change_epoch2:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

def _label_smoothing(label, num_class=10, factor=0.1):
    one_hot = np.eye(num_class)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_class - 1))

    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

def train_adversarial(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()

        benign_outputs = net(adv_inputs)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            loss = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()
    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

def train(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer, config: Any) -> Tuple[float, float]:
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc='>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        benign_outputs = net(inputs)

        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            c_ls = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            c_ls = criterion(benign_outputs, targets)
        c_ls.backward()

        optimizer.step()
        train_loss += c_ls.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2), loss=c_ls.item())
        train_bar.update(1)
    train_bar.close()

    return 100. * correct / total, train_loss

def test_net_normal(net: nn.Module, test_loader: DataLoader, epoch: int, optimizer: Optimizer, 
         best_prec: float, config: Any,save_path='./checkpoint',) -> Tuple[float, float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    is_best = test_acc > best_prec
    best_prec_robust = max(test_acc, best_prec)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec_robust,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(save_path))
    print('Model Saved!')
    return test_acc, adv_acc, benign_loss_test, best_prec_robust

def test_net_robust(net: nn.Module, test_loader: DataLoader, epoch: int, optimizer: Optimizer, 
         best_prec: float, config: Any,save_path='./checkpoint',) -> Tuple[float, float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    is_best = adv_acc > best_prec
    best_prec_robust = max(adv_acc, best_prec)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec_robust,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(save_path))
    print('Model Saved!')
    return test_acc, adv_acc, benign_loss_test, best_prec_robust

# PGD attack
def pgd_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

# PGD attack
def pgd_attack_TRADES(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss_1 = F.kl_div(F.log_softmax(logits, dim=1),
                                F.softmax(model(x), dim=1),
                                reduction='batchmean')
        grad = torch.autograd.grad(loss_1, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def pgd_invserse_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def pgd_invserse_attack_UIAT(model: nn.Module, x: Tensor, x_attack: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()
    logits_natural = model(x)

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y) + F.kl_div(F.log_softmax(logits, dim=1),
                                F.softmax(logits_natural, dim=1),
                                reduction='batchmean') - F.kl_div(F.log_softmax(logits, dim=1),
                                F.softmax(model(x_attack), dim=1),
                                reduction='batchmean')
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def test_pgd(net: nn.Module, test_loader: DataLoader, config: Any) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                         config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    print('\n---->PGD attack test accuarcy:', adv_acc)
    return adv_acc

def test_net(net: nn.Module, test_loader: DataLoader, config: Any) -> Tuple[float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    return test_acc, adv_acc, benign_loss_test

def val_net(net: nn.Module, epoch: int, val_loader: DataLoader, optimizer: Optimizer, 
         best_val_robust_acc: float, config: Any, check_path='./checkpoint') -> Tuple[float, float, float]:
    benign_loss_val = 0
    val_benign_correct = 0
    val_adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., 10)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_val += loss.item()
        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        val_adv_correct += predicted_adv.eq(targets).sum().item()
        val_benign_correct += predicted.eq(targets).sum().item()
    val_test_acc = 100. * val_benign_correct / total
    val_adv_acc = 100. * val_adv_correct / total
    is_best = (val_adv_acc > best_val_robust_acc)
    best_val_robust_acc = max(val_adv_acc, best_val_robust_acc)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_val_robust_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(check_path))
    return val_test_acc, val_adv_acc, best_val_robust_acc

def record_path_words(record_path, record_words):
    print(record_words)
    with open(record_path, "a+") as f:
        f.write(record_words)
    f.close()
    return

def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def cw_Linf_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = CW_loss(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()

        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def train_adversarial_CFA(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = cw_Linf_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        optimizer.zero_grad()
        adv_outputs = net(adv_inputs)
        loss_logits = criterion(adv_outputs, targets)
        loss = loss_logits
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = net(adv_inputs).max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss