import torch
import torch.nn as nn
import torchattacks
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from autoattack import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_autoattack(net: nn.Module, test_loader: DataLoader, eps: int, attacks_run: list) -> float:
    net.eval()

    autoattack = AutoAttack(net, norm='Linf', eps=eps/255., seed=1,
                            attacks_to_run=attacks_run, version='custom')
    autoattack.apgd.n_restarts = 2
    autoattack.fab.n_restarts = 2
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    #TODO: Modify return value of run_standard_evaluation
    x_adv, robust_accuracy = autoattack.run_standard_evaluation((x_test).to(device), y_test.to(device))

    print(f"Autoattack have done! Accruracy{robust_accuracy*100.:.2f}")

    return robust_accuracy*100.

def evaluate_autoattack_l2(net: nn.Module, test_loader: DataLoader, eps: int, attacks_run: list) -> float:
    net.eval()

    autoattack = AutoAttack(net, norm='L2', eps=eps/255., seed=1,
                            attacks_to_run=attacks_run, version='custom')
    autoattack.apgd.n_restarts = 2
    autoattack.fab.n_restarts = 2
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    #TODO: Modify return value of run_standard_evaluation
    x_adv, robust_accuracy = autoattack.run_standard_evaluation((x_test).to(device), y_test.to(device))

    print(f"Autoattack have done! Accruracy{robust_accuracy*100.:.2f}")

    return robust_accuracy*100.