from .. import *
from ..utils.misc import *
import torch.nn.functional as F

class LeNet3(nn.Module):          
    '''
    two convolutional layers of sizes 64 and 128, and a fully connected layer of size 1024
    suggested by 'Adversarial Robustness vs. Model Compression, or Both?'
    '''
    def __init__(self, data_code='mnist', **kwargs): 
        
        super(LeNet3, self).__init__()
        if 'robustness' in kwargs:
            self.rob = kwargs['robustness']
        else:
            self.rob = False
            
        in_ch = get_in_channels(data_code)
        
        self.conv1 = torch.nn.Conv2d(in_ch, 32, 5, 1, 2) # in_channels, out_channels, kernel, stride, padding
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1, 2)
        
        # Fully connected layer
        if data_code == 'mnist':
            dim = 7*7*64
        elif data_code == 'cifar10':
            dim = 8*8*64
            
        self.fc1 = torch.nn.Linear(dim, 1024)   # convert matrix with 400 features to a matrix of 1024 features (columns)
        self.fc2 = torch.nn.Linear(1024, 10)
        
    def forward(self, x):
        output_list = []
        
        x = F.relu(self.conv1(x))
        output_list.append(x)
        x = F.max_pool2d(x, 2, 2)
        #output_list.append(x)
        x = F.relu(self.conv2(x))
        output_list.append(x)
        x = F.max_pool2d(x, 2, 2)
        #output_list.append(x)
        
        x = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(x))
        output_list.append(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        #output_list.append(x)
        if self.rob:
            return x
        else:
            return x, output_list
        
class LeNet4(nn.Module):          
    '''
    two convolutional layers of sizes 64 and 128, and a fully connected layer of size 1024
    suggested by 'Adversarial Robustness vs. Model Compression, or Both?'
    '''
    def __init__(self, data_code='cifar10', **kwargs): 
        
        super(LeNet4, self).__init__()
        if 'robustness' in kwargs:
            self.rob = kwargs['robustness']
        else:
            self.rob = False
            
        in_ch = get_in_channels(data_code)
        
        self.conv1 = torch.nn.Conv2d(in_ch, 96, 5) # in_channels, out_channels, kernel, stride, padding
        self.conv2 = torch.nn.Conv2d(96, 256, 5)
        
        # Fully connected layer
        dim = 6400
            
        self.fc1 = torch.nn.Linear(6400,1920)
        self.fc2 = torch.nn.Linear(1920,1344)
        self.fc3 = torch.nn.Linear(1344, 10)
        
    def forward(self, x):
        output_list = []
        
        x = F.relu(self.conv1(x))
        output_list.append(x)
        x = F.max_pool2d(x, 2)
        #output_list.append(x)
        x = F.relu(self.conv2(x))
        output_list.append(x)
        x = F.max_pool2d(x, 2)
        #output_list.append(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output_list.append(x)
        x = F.relu(self.fc2(x))
        output_list.append(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        #output_list.append(x)
        if self.rob:
            return x
        else:
            return x, output_list