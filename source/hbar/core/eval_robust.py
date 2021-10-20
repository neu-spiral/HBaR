import torchattacks
from .. import *
from .  import *

from .train_misc     import *
from .train_hsic     import *
from .train_standard import *
from ..utils.path    import *

def eval_robust(config_dict):
    level_robust_list, level_prec1_list, prec1_list, rob1_list = [], [], [], []
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])
    batch_acc = meter.AverageMeter()
    batch_acc_top5 = meter.AverageMeter()
    batch_rob = meter.AverageMeter()
    
    model = model_distribution(config_dict)
    copynet = load_model(get_model_path("{}".format(config_dict['model_file'])))
    model.load_state_dict(copynet)
    model.to(config_dict['device'])
    model.eval()
    
    #print(config_dict['epsilon'],config_dict['pgd_alpha'],config_dict['pgd_steps'])
    
    eps = config_dict['epsilon']
    alpha = config_dict['pgd_alpha']
    print(config_dict['attack_type'], eps, alpha, config_dict['pgd_steps'])
    
    if config_dict['attack_type'] == 'pgd':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=config_dict['pgd_steps'])
    elif config_dict['attack_type'] == 'fgsm':
        attack = torchattacks.FGSM(model, eps=eps)
        
    n_data = config_dict['batch_size'] * len(test_loader)
    pbar = tqdm(enumerate(test_loader), total=n_data/config_dict['batch_size'], ncols=120)

    softmax = torch.nn.Softmax()
    #print(alpha, steps, eps)
    
    for batch_idx, (data, target) in pbar:
        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])

        data_attacked = attack(data, target)
        output_attacked = model(data_attacked)
        output = model(data)

        #output, output_attacked = softmax(output), softmax(output_attacked)
        level_robust = torch.dist(output, output_attacked).cpu().detach().numpy()
        level_prec1, level_prec5 = get_accuracy(output, torch.argmax(output_attacked, dim=1), topk=(1, 5))

        prec1, prec5 = get_accuracy(output, target, topk=(1, 5))
        rob1, rob5 = get_accuracy(output_attacked, target, topk=(1, 5))

        level_robust_list.append(level_robust)
        level_prec1_list.append(level_prec1.cpu().detach().numpy())
        prec1_list.append(prec1.cpu().detach().numpy())
        rob1_list.append(rob1.cpu().detach().numpy())

    print("Test Accuracy: {:.4f}; Adv Accuracy: {:.4f}; Level Accuracy: {:.4f}; Level Dist: {:.4f}".format(np.mean(prec1_list), np.mean(rob1_list), np.mean(level_prec1_list), np.mean(level_robust_list)))
    #print(np.mean(prec1_list))