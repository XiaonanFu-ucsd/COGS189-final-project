import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import pickle

# Define our filter variables
fs = 512                      # Hz; sampling rate
dt = 1000. / fs                 # ms; time between samples
sdt = dt#np.round(dt).astype(int); # rounded dt so that we can index samples
hp = 1                        # Hz; our low cut for our bandpass
lp = 24.                        # Hz; our high cut for our bandpass
num_taps = 31                   # Number of taps/coefficients of FIR filter

# Define ERP-related variables
epoch_start = 0    # ms
epoch_end = 800    # ms
baseline_start = 0 # ms
baseline_end = 100 # ms
erp_start = 200    # ms
erp_end = 800      # ms

# Let's translate these from time into index space to save time later
e_s = np.round(epoch_start / sdt).astype(int)     # epoch start
e_e = np.round(epoch_end / sdt).astype(int)       # epoch end
bl_s = np.round(baseline_start / sdt).astype(int) # baseline start
bl_e = np.round(baseline_end / sdt).astype(int)   # baseline end
erp_s = np.round(erp_start / sdt).astype(int)     # ERP component window start
erp_e = np.round(erp_end / sdt).astype(int)       # ERP component window end


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class EEGNet(nn.Module):
    # expected input size: (batch_size, 32, 1, 128)
    def __init__(self, time_length, num_chans, output_size, dropout=0.3):
        super(EEGNet, self).__init__()
        self.time_length = time_length
        self.num_chans = num_chans
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_chans, 16, (1, 64), padding = 'same'),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, groups=8),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            SeparableConv2d(32, 64, (1, 32), padding = 'same'),
            nn.BatchNorm2d(64),
            # nn.Conv2d(256, 32, (1, 4), padding='same'),
            # nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        # reshape to (batch_size, 32, 1, 128)
        x = torch.reshape(x, (x.shape[0], self.num_chans, 1, self.time_length))
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x




def load_model(model_path, device):
    if model_path is None:
        model = EEGNet(128, 32, 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, opt
    with open(model_path, 'rb') as f:
        ckpoint = torch.load(f)
        model = EEGNet(128, 32, 2).to(device)
        model.load_state_dict(ckpoint['state_dict'])
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        opt.load_state_dict(ckpoint['optimizer'])
    return model, opt

def test(
    X_test
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, opt = load_model(None, device)
    net.to(device)
    class_weights = [1, 5.4]
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    # preprocess
    def preprocess(x):
        for i in range(x.shape[0]):
            # correct DC offset of signal
            x[i] = x[i] - np.mean(x[i], axis=1).reshape(-1, 1)
            # devide sd from every channel
            sd_every_chan = np.std(x[i], axis=1).reshape(-1, 1)
            x[i] = x[i] / sd_every_chan
    preprocess(X_test)

    # downsample
    num_points = 128; # we will divide our window into num_points means
    # Define a simple windowed means function
    def wm(x, start, end, num_points):
        num_trials = x.shape[0] # assumes first dem is numb observations
        num_chans = x.shape[1] # assumes last dim is num channels
        len_time = x.shape[2] # assumes second dim is time
        w = np.round((end-start)/num_points)
        y = np.zeros((num_trials, num_chans, num_points))
        for i in range(0, num_points):
            s = (start + (w * i))
            e = (s + w)
            if e > len_time:
                e = len_time
            y[:,:,i] = np.mean(x[:,:,s.astype(int):e.astype(int)], axis=2)
        return y

    
    X_test = wm(X_test, erp_s, erp_e, num_points).astype(np.float32)
    
    # transpose each epoch to (128, 32)
    tmp_x = []
    for i in range(X_test.shape[0]):
        tmp_x.append(np.transpose(X_test[i]))
    X_test = np.array(tmp_x)
    del tmp_x
    
    testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test))
    testloader = DataLoader(trainset, batch_size=256, shuffle=False)
    
    # get predictions
    correct = 0
    total = 0
    net.eval()
    y_pred = np.array([])
    y_gt = np.array([])
    with torch.no_grad():
        for data in trainloader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())
            #y_gt = np.append(y_gt, y.cpu().numpy())
    y_pred[y_pred == 0] = -1
    return y_pred


def _test(
    net,
    opt,
    X_test
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    class_weights = [1, 5.4]
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    # preprocess
    def preprocess(x):
        for i in range(x.shape[0]):
            # correct DC offset of signal
            x[i] = x[i] - np.mean(x[i], axis=1).reshape(-1, 1)
            # devide sd from every channel
            sd_every_chan = np.std(x[i], axis=1).reshape(-1, 1)
            x[i] = x[i] / sd_every_chan
    preprocess(X_test)

    # downsample
    num_points = 128; # we will divide our window into num_points means
    # Define a simple windowed means function
    def wm(x, start, end, num_points):
        num_trials = x.shape[0] # assumes first dem is numb observations
        num_chans = x.shape[1] # assumes last dim is num channels
        len_time = x.shape[2] # assumes second dim is time
        w = np.round((end-start)/num_points)
        y = np.zeros((num_trials, num_chans, num_points))
        for i in range(0, num_points):
            s = (start + (w * i))
            e = (s + w)
            if e > len_time:
                e = len_time
            y[:,:,i] = np.mean(x[:,:,s.astype(int):e.astype(int)], axis=2)
        return y

    
    X_test = wm(X_test, erp_s, erp_e, num_points).astype(np.float32)
    
    # transpose each epoch to (128, 32)
    tmp_x = []
    for i in range(X_test.shape[0]):
        tmp_x.append(np.transpose(X_test[i]))
    X_test = np.array(tmp_x)
    del tmp_x
    
    testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test))
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    
    # get predictions
    correct = 0
    total = 0
    net.eval()
    y_pred = np.array([])
    with torch.no_grad():
        for data in testloader:
            x = data[0]
            x = x.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())
    y_pred[y_pred == 0] = -1
    return y_pred





def train_and_test(
    model_path,
    X_train_,
    y_train_,
    X_test
):
    
    if type(X_train_) == type(None):
        return test(X_test)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, opt = load_model(model_path, device)
    
    net.to(device)
    class_weights = [1, 5.4]
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    X_train = X_train_.copy()
    y_train = y_train_.copy()
    y_train[y_train == -1] = 0
    
    def preprocess(x):
        for i in range(x.shape[0]):
            # correct DC offset of signal
            x[i] = x[i] - np.mean(x[i], axis=1).reshape(-1, 1)
            # devide sd from every channel
            sd_every_chan = np.std(x[i], axis=1).reshape(-1, 1)
            x[i] = x[i] / sd_every_chan
    preprocess(X_train)
    
    num_points = 128; # we will divide our window into num_points means
    # Define a simple windowed means function
    def wm(x, start, end, num_points):
        num_trials = x.shape[0] # assumes first dem is numb observations
        num_chans = x.shape[1] # assumes last dim is num channels
        len_time = x.shape[2] # assumes second dim is time
        w = np.round((end-start)/num_points)
        y = np.zeros((num_trials, num_chans, num_points))
        for i in range(0, num_points):
            s = (start + (w * i))
            e = (s + w)
            if e > len_time:
                e = len_time
            y[:,:,i] = np.mean(x[:,:,s.astype(int):e.astype(int)], axis=2)
        return y

    X_train = wm(X_train, erp_s, erp_e, num_points).astype(np.float32)
    
    # transpose each epoch to (128, 32)
    tmp_x = []
    for i in range(X_train.shape[0]):
        tmp_x.append(np.transpose(X_train[i]))
    X_train = np.array(tmp_x)
    del tmp_x

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True)
    

    avg_losses = []   # Avg. losses.
    epochs = min(30, int(2000000/trainloader.dataset.tensors[0].shape[0]))       # Total epochs.
    print('Total epochs: {}'.format(epochs))
    print_freq = 30  # Print frequency.

    iter_count = 0

    net.train()
    for epoch in range(epochs):  # Loop over the dataset multiple times.
        running_loss = 0.0       # Initialize running loss.
        for i, data in enumerate(trainloader, 0):
            # Get the inputs.
            inputs, labels = data
            
            # Move the inputs to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients.
            opt.zero_grad()

            # Forward step.
            outputs = net(inputs)
            labels = labels.reshape(-1,).long()
            loss = loss_func(outputs, labels)
            
            # Backward step.
            loss.backward()
            
            # Optimization step (update the parameters).
            opt.step()

            # Print statistics.
            running_loss += loss.item()
            if iter_count % print_freq == print_freq - 1: # Print every several mini-batches.
                avg_loss = running_loss / print_freq
                avg_acc = (outputs.argmax(dim=1) == labels).float().mean()
                
                print('[epoch: {}, i: {:5d}] avg mini-batch loss: {:.3f} avg mini-batch acc: {:.3f}'.format(
                    epoch, i, avg_loss, avg_acc))
                avg_losses.append(avg_loss)
                running_loss = 0.0
                
            iter_count += 1
    
    # predict
    return _test(net, opt, X_test)