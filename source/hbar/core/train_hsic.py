from .. import *
from .  import *
from .train_misc     import *

def hsic_train(cepoch, model, data_loader, optimizer, scheduler, config_dict):

    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = meter.AverageMeter()
    batch_loss   = meter.AverageMeter()
    batch_hischx = meter.AverageMeter()
    batch_hischy = meter.AverageMeter()

    model = model.to(config_dict['device'])
    model.train()
    
    batch_size = config_dict['batch_size']
    n_data = batch_size * len(data_loader)

    if config_dict['adv_train']:
        attack = torchattacks.PGD(model, eps=config_dict['epsilon'], \
                                  alpha=config_dict['pgd_alpha'], steps=config_dict['pgd_steps'])
        
    pbar = tqdm(enumerate(data_loader), total=n_data/batch_size, ncols=120)
    for batch_idx, (data, target) in pbar:
        scheduler.step()
        
        # data augmentation for robustness attack
        if config_dict['aug_data']:
            attack = np.random.normal(config_dict['epsilon']/2, config_dict['epsilon']/6, size=data.shape)
            data += attack
            
        data   = data.float().to(config_dict['device'])
        target = target.to(config_dict['device'])
        total_loss = 0
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
        total_loss += (loss * config_dict['xentropy_weight'])
        
        # compute hsic
        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=10).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))

        # new variable
        hx_l_list = []
        hy_l_list = []
        lx, ly, ld = config_dict['lambda_x'], config_dict['lambda_y'], config_dict['hsic_layer_decay']
        if ld > 0:
            lx, ly = lx * (ld ** len(hiddens)), ly * (ld ** len(hiddens))
            
        for i in range(len(hiddens)):
            
            if len(hiddens[i].size()) > 2:
                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

            hx_l, hy_l = hsic_objective(
                    hiddens[i],
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=config_dict['sigma'],
                    k_type_y=config_dict['k_type_y']
            )

            hx_l_list.append(hx_l)
            hy_l_list.append(hy_l)
            
            if ld > 0:
                lx, ly = lx/ld, ly/ld
                #print(i, lx, ly)
            temp_hsic = lx * hx_l - ly * hy_l
            total_loss += temp_hsic.to(config_dict['device'])
                         
        total_loss.backward() # Back Propagation
        optimizer.step() # Gradient Descent

        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 

        batch_acc.update(prec1)
        batch_loss.update(float(loss.detach().cpu().numpy())) # this is just for xentropy loss! total loss is for xentropy + hsic
        batch_hischx.update(sum(hx_l_list).cpu().detach().numpy())
        batch_hischy.update(sum(hy_l_list).cpu().detach().numpy())
        
        # # # preparation log information and print progress # # #

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