import collections

from .. import *
from .  import *

from .train_misc     import *
from .train_hsic     import *
from .train_standard import *
from ..utils.path    import *

def training_standard(config_dict):
    """
    Zifeng : Train *all* model (hsic + vanilla) parameters with cross-entropy loss
    """
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])

    torch.manual_seed(config_dict['seed'])

    model = model_distribution(config_dict)
    config_dict['robustness'] = True
    model_single_output = model_distribution(config_dict)
    for name, weight in model.named_parameters():
        print(name)
        
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader)

    log_dict = {}
    batch_log_list = []
    epoch_log_dict = collections.defaultdict(list)

    nepoch = config_dict['epochs']

    for cepoch in range(1, nepoch+1):
        if config_dict['training_type'] == 'hsictrain':
            hsic_train(cepoch, model, train_loader, optimizer, scheduler, config_dict)
        elif config_dict['training_type'] == 'backprop':
            standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict)
        else:
            raise ValueError("Unknown training type or not support [{}]".format(config_dict['training_type']))
            
        train_loss, train_acc, train_hx, train_hy = misc.get_hsic_epoch(config_dict, model, train_loader)
        test_loss, test_acc, test_hx, test_hy = misc.get_hsic_epoch(config_dict, model, test_loader)
        epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
        print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}.".format(cepoch, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy))

        if config_dict['save_last_model_only'] and cepoch == nepoch:
            filename = os.path.splitext(config_dict['model_file'])[0]
            save_model(model,get_model_path("{}.pt".format(filename)))
    
        # Robustness Analysis
        model_single_output.load_state_dict(model.state_dict())
        model_single_output.eval()
        rob_acc, rob_acc5 = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
        epoch_log_dict['rob'].append(rob_acc)
        
    log_dict['epoch_log_dict'] = epoch_log_dict
    log_dict['config_dict'] = config_dict
    filename = "{}.npy".format(os.path.splitext(config_dict['model_file'])[0])
    save_logs(log_dict, get_log_filepath("{}".format(filename)))

    return batch_log_list, epoch_log_dict