from .. import *
from ..math.hsic import *
import torchattacks

def get_current_timestamp():
    return strftime("%y%m%d_%H%M%S")

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'cifar10':
        in_ch = 3
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    in_dim = -1    
    if data_code == 'mnist':
        in_dim = 784
    elif data_code == 'cifar10':
        in_dim = 1024
    elif data_code == 'fmnist':
        in_dim = 784
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_dim

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        output, hiddens = model(data)
        loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
        acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    return np.mean(acc), np.mean(loss)

def get_hsic_epoch(config_dict, model, dataloader):
    """ Computes the hsic
    """
    acc = []
    loss = []
    hx_l_list = []
    hy_l_list = []
    model = model.to('cuda')
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            output, hiddens = model(data)

            # compute acc
            acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())

            # compute hsic
            h_target = target.view(-1,1)
            h_target = to_categorical(h_target, num_classes=10).float()
            h_data = data.view(-1, np.prod(data.size()[1:]))

            hsic_hx = hsic_normalized_cca( hiddens[-1], h_data,   sigma=config_dict['sigma'])
            hsic_hy = hsic_normalized_cca( hiddens[-1], h_target, sigma=config_dict['sigma'], k_type_y=config_dict['k_type_y'])
            hsic_hx = hsic_hx.cpu().detach().numpy()
            hsic_hy = hsic_hy.cpu().detach().numpy()

            hx_l_list.append(hsic_hx)
            hy_l_list.append(hsic_hy)

            # compute loss
            loss.append(torch.nn.CrossEntropyLoss()(output, target).cpu().detach().numpy())
    return np.mean(loss), np.mean(acc), np.mean(hx_l_list), np.mean(hy_l_list)


def eval_robust_epoch(model, dataloader, config_dict):
    acc = []
    acc5 = []
    model = model.to('cuda')
    device = next(model.parameters()).device
    model.eval()
    
    eps = config_dict['epsilon']
    alpha = config_dict['pgd_alpha']
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=config_dict['pgd_steps'])
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        data_attacked = attack(data, target)
        output = model(data_attacked)

        prec1, prec5 = get_accuracy(output, target, topk=(1, 5)) 
        acc.append(prec1.cpu().detach().numpy())
        acc5.append(prec5.cpu().detach().numpy())
    print("Average robust accuracy is top1: {:.4f}, top5: {:.4f}".format(np.mean(acc), np.mean(acc5)))
    return np.mean(acc), np.mean(acc5)

def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy):
    epoch_log_dict['train_loss'].append(train_loss)
    epoch_log_dict['train_acc'].append(train_acc)
    epoch_log_dict['train_hx'].append(train_hx)
    epoch_log_dict['train_hy'].append(train_hy)

    epoch_log_dict['test_loss'].append(test_loss)
    epoch_log_dict['test_acc'].append(test_acc)
    epoch_log_dict['test_hx'].append(test_hx)
    epoch_log_dict['test_hy'].append(test_hy)
    return epoch_log_dict

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])
