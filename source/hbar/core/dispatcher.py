from .. import *
from .engine import *

def job_execution(config_dict):

    torch.cuda.manual_seed(config_dict['seed'])
    torch.manual_seed(config_dict['seed'])
    
    if config_dict['task'] == 'pre-train':
        if config_dict['do_training']:
            out_batch, out_epoch = training_standard(config_dict)
            
    else:
        raise ValueError("Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict['task']))
        
    