import torch.backends.cudnn as cudnn
from easydict import EasyDict
import yaml
import logging
from models import *
from utils_train import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inverse_pgd_attack(model: nn.Module, x: Tensor, y: Tensor, x_attack: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv_inverse = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv_inverse = torch.clamp(x_adv_inverse, 0, 1)
    criterion = nn.CrossEntropyLoss()
    logits_natural = model(x)
    logits_attack = model(x_attack)

    for _ in range(iters):
        x_adv_inverse.requires_grad = True
        logits_inverse_x = model(x_adv_inverse)
        loss = criterion(logits_inverse_x, y) + F.kl_div(F.log_softmax(logits_inverse_x, dim=1), \
                                               F.softmax(logits_natural, dim=1), reduction='batchmean') \
                                    - F.kl_div(F.log_softmax(logits_inverse_x, dim=1), \
                                               F.softmax(logits_attack, dim=1), reduction='batchmean')
        grad = torch.autograd.grad(loss, x_adv_inverse)[0]
        perturbation = alpha * torch.sign(grad.detach())
        x_adv_inverse = x_adv_inverse.detach() - perturbation
        x_adv_inverse = torch.min(torch.max(x_adv_inverse, x - epsilon), x + epsilon)
        x_adv_inverse = torch.clamp(x_adv_inverse, 0, 1)

    return x_adv_inverse.detach()

def pgd_attack_union(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv_inverse = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv_inverse = torch.clamp(x_adv_inverse, 0, 1)
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()
    logits_natural = model(x)

    for _ in range(iters):
        x_adv_inverse.requires_grad = True
        x_adv.requires_grad = True
        logits_inverse_x = model(x_adv_inverse)
        logits_x = model(x_adv)
        loss_ix = criterion(logits_inverse_x, y) \
                                    + F.kl_div(F.log_softmax(logits_inverse_x, dim=1), \
                                               F.softmax(logits_natural, dim=1), reduction='batchmean') \
                                    - F.kl_div(F.log_softmax(logits_inverse_x, dim=1), \
                                               F.softmax(logits_x, dim=1), reduction='batchmean')
        loss_x = criterion(logits_x, y)
        grad_inverse_attack = torch.autograd.grad(loss_ix, x_adv_inverse)[0]
        x_adv_inverse = x_adv_inverse.detach() - alpha * torch.sign(grad_inverse_attack.detach())
        x_adv_inverse = torch.min(torch.max(x_adv_inverse, x - epsilon), x + epsilon)
        x_adv_inverse = torch.clamp(x_adv_inverse, 0, 1)

        grad_attack = torch.autograd.grad(loss_x, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad_attack.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv_inverse.detach(), x_adv.detach()

def Mininal_vector_orthogonalization(x1, x2):
    #TODO: compute_projection of x1 and x2
    projection = torch.matmul(x1, x2.t()) / torch.norm(torch.matmul(x2, x2.t()),p=2)
    return torch.norm(x1-torch.matmul(projection, x2),p=2)/x1.shape[0]

def _label_smoothing(label, num_class=10, factor=0.1):
    one_hot = np.eye(num_class)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_class - 1))

    return result

def train_adversarial_Segment(net: nn.Module, Segment_net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
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
        target_categories = np.array(targets.cpu())
        targets_segment = [ClassifierOutputTarget(
            category) for category in target_categories]
        Grad_segment_value = Segment_net(input_tensor=inputs,
                targets=targets_segment, 
                eigen_smooth=False)
        Sign_mask = (Grad_segment_value>config.Segmentation.Threshold).astype(np.float32)
        Sign_tensor = torch.Tensor(Sign_mask).unsqueeze(1).to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        adv_invserse_inputs = inverse_pgd_attack(net, inputs, targets, adv_inputs, config.Train.inverse_clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        optimizer.zero_grad()
        benign_outputs = net(adv_inputs)
        aug = benign_outputs
        logits_inverse = net(adv_invserse_inputs)
        logits_inverse_s = net(adv_invserse_inputs*(1-Sign_tensor))
        momentum = 1.0
        logit_inverse_refine = logits_inverse - logits_inverse_s*momentum
        
        loss_kl = F.kl_div((F.softmax(logit_inverse_refine, dim=1)  + 1e-12).log(), F.softmax(aug, dim=1) + 1e-12, reduction='batchmean')
        loss_ortho = Mininal_vector_orthogonalization(logits_inverse, logits_inverse_s)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            loss_CE = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            loss_CE = criterion(benign_outputs, targets)
        loss = loss_CE + loss_kl+ loss_ortho
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

if __name__=='__main__':
    with open('configs_train.yml') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    net = ResNet18()
    net_CAM = ResNet18()

    file_name = config.Operation.Prefix
    data_set = config.Train.Data
    check_path = os.path.join('./checkpoint', data_set, file_name)
    learning_rate = config.Train.Lr

    if not os.path.isdir(os.path.join('./checkpoint', data_set)):
        os.mkdir(os.path.join('./checkpoint', data_set))
    if not os.path.isdir(check_path):
        os.mkdir(check_path)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(check_path, file_name + '_record.log')),
            logging.StreamHandler()
        ])

    net.num_classes = config.DATA.num_class
    norm_mean = torch.tensor(config.DATA.mean).to(device)
    norm_std = torch.tensor(config.DATA.std).to(device)

    net.norm = True
    net.mean = norm_mean
    net.std = norm_std
    
    net_CAM.num_classes = config.DATA.num_class
    net_CAM.norm = True
    net_CAM.mean = norm_mean
    net_CAM.std = norm_std
    
    Data_norm = False
    logger.info('Adversarial Training || net: '+config.Operation.Prefix + ' || '+config.Train.Train_Method)

    train_loader, test_loader = create_dataloader(data_set, Norm=Data_norm)

    net = net.to(device)
    net = torch.nn.DataParallel(net)  # parallel GPU
    cudnn.benchmark = True

    #TODO: Grad_CAM model
    net_CAM = net_CAM.to(device)
    net_CAM = torch.nn.DataParallel(net_CAM)
    check_path_CAM = os.path.join('./checkpoint', data_set, config.Segmentation.Segmentation_Model)
    checkpoint_CAM = torch.load(os.path.join(check_path_CAM, 'checkpoint.pth.tar'), map_location=device)
    net_CAM.load_state_dict(checkpoint_CAM['state_dict'])

    Grad_segment = GradCAM(net_CAM, [net_CAM.module.layer4])

    if config.Operation.Resume == True:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
    else:
        start_epoch = 0
        best_prec1 = 0
        logger.info(config.Operation.record_words)
        logger.info('%-5s\t%-10s\t%-9s\t%-9s\t%-8s\t%-15s', 'Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Test Robust Acc')

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch + 1, config.Train.Epoch + 1):
        learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch, config.Train.lr_change_iter[0], config.Train.lr_change_iter[1])
        acc_train, train_loss  = train_adversarial_Segment(net, Grad_segment, epoch,  train_loader, optimizer, config)
        acc_test, pgd_acc, loss_test, best_prec1 = test_net_robust(net, test_loader, epoch, optimizer, best_prec1, config, save_path=check_path)
        logger.info('%-5d\t%-10.2f\t%-9.2f\t%-9.2f\t%-8.2f\t%.2f', epoch, train_loss, acc_train, loss_test,
                    acc_test, pgd_acc) 

