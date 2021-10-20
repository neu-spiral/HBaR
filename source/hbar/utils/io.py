from .. import *
from .misc  import *
from .path  import *
import yaml

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return data
    
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    
def load_model(filepath):
    model = torch.load(filepath)
    return model

def save_logs(logs, filepath):
    np.save(filepath, logs)
    
def load_logs(filepath):
    logs = np.load(filepath, allow_pickle=True)[()]
    return logs

