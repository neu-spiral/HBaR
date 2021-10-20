from .. import *
from .  import *
from .train_misc     import *

batch_acc    = meter.AverageMeter()
batch_loss   = meter.AverageMeter()
batch_hischx = meter.AverageMeter()
batch_hischy = meter.AverageMeter()

def standard_train(cepoch, model, data_loader, optimizer, scheduler, config_dict):

    prec1 = total_loss = hx_l = hy_l = -1

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)
    
    if config_dict['adv_train']:
        attack = torchattacks.PGD(model, eps=config_dict['epsilon'], \
                                  alpha=config_dict['pgd_alpha'], steps=config_dict['pgd_steps'])
    
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=150)
    # for batch_idx, (data, target) in enumerate(data_loader):
    for batch_idx, (data, target) in pbar:

        if os.environ.get('HSICBT_DEBUG')=='4':
            if batch_idx > 5:
                break
        
        # adjust learning rate
        scheduler.step()
            
        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        optimizer.zero_grad()
        
        # several tricks
        if config_dict['mixup']:
            data, target_a, target_b, lam = mixup_data(data, target, config_dict['alpha'])
        
        # adversarial training
        if config_dict['adv_train']:
            attacked_data = attack(data, target)
            output, hiddens = model(attacked_data)
            _, hiddens = model(data)
        else:
            output, hiddens = model(data)

        # compute loss
        criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config_dict['smooth_eps']).to(config_dict['device'])
        if config_dict['mixup']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, config_dict['smooth'])
        else:
            loss = criterion(output, target, smooth=config_dict['smooth'])
        
        # compute hsic
        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=10).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))
        hx_l, hy_l = hsic_objective(
                hiddens[-1],
                h_target=h_target.float(),
                h_data=h_data,
                sigma=5,
                k_type_y=config_dict['k_type_y']
            )
        hx_l = hx_l.cpu().detach().numpy()
        hy_l = hy_l.cpu().detach().numpy()
        
        # add l1_norm
        if 'l1_norm' in config_dict and config_dict['l1_norm']:
            for hidden in hiddens:
                loss += config_dict['l1_weight']*torch.norm(hidden, 1) 
            
        ### Tong: add admm
        if ADMM is not None:
            z_u_update(config_dict, ADMM, model, cepoch, batch_idx)  # update Z and U variables
            pure_loss, admm_loss, loss = append_admm_loss(ADMM, model, loss)  # append admm losses
        
        loss.backward()
        optimizer.step()


        loss = float(loss.detach().cpu().numpy())
        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 
        prec1 = float(prec1.cpu().numpy())
    
        batch_acc.update(prec1)   
        batch_loss.update(loss)  
        batch_hischx.update(hx_l)
        batch_hischy.update(hy_l)

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )

        pbar.set_description(msg)
