# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import glob
import time
import os
# import torchmetrics
import matplotlib.pyplot as plt
import pickle
# from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
# from torchvision.datasets import ImageFolder
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
from PIL import Image

# %%
# part A : Batch Normalisation
class BatchNorm2d(nn.Module): #My definition of Batch Normalisation.
    def __init__(self, size):
        super(BatchNorm2d, self).__init__()
        self.epsilon = 1e-5;
        shape = (1, size, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape)) # the scaling factor that determines the new standard deviation.
        self.beta = nn.Parameter(torch.zeros(shape)) # the bias that is the new mean.
        self.running_sum = torch.zeros(shape).to(device);
        self.running_square_sum = torch.zeros(shape).to(device);
        self.total = 0;
    def forward(self, X):
        #if we are in training mode, then we use the mean and variance of this batch.
        if self.training:
            mean = torch.mean(X, dim = (0,2,3), keepdim = True);
            var = torch.var(X,dim = (0,2,3), keepdim = True);
            self.total += 1;
            self.running_sum += mean;
            self.running_square_sum += var;
        else:
            mean = self.running_sum / self.total;
            var = self.running_square_sum / self.total;
        # X_mean = torch.ones(X.shape) * mean;
#         X_mean = mean.expand_as(X);
        X_transformed = (X - mean) / torch.sqrt(var + self.epsilon); #epsilon is added for non-zero denominator
        # X_transformed = self.gamma * X_transformed + self.beta;
        X_transformed = X_transformed * self.gamma + self.beta;
        return X_transformed;
class InstanceNormalisation2d(nn.Module):
    def __init__(self, size):
        super(InstanceNormalisation2d, self).__init__();
        self.epsilon = 1e-5;
        self.gamma = nn.Parameter(torch.ones((1, size, 1, 1)));
        self.beta = nn.Parameter(torch.zeros((1, size, 1, 1)));
    def forward(self, X):
        mean = torch.mean(X, dim = (2,3), keepdim = True);
        var = torch.var(X, dim = (2,3), keepdim = True);
        X_transformed = (X - mean) / torch.sqrt(var + self.epsilon);
        X_transformed = X_transformed * self.gamma + self.beta;
        return X_transformed;
class BatchInstanceNormalisation2d(nn.Module):
    def __init__(self, size):
        super(BatchInstanceNormalisation2d, self).__init__();
        self.batch_norm = BatchNorm2d(size);
        self.instance_norm = InstanceNormalisation2d(size);
        shape = (1, size, 1, 1)
        self.rho = nn.Parameter(torch.ones(shape));
        self.epsilon = 1e-5;
        self.gamma = nn.Parameter(torch.ones(shape)) # the scaling factor that determines the new standard deviation.
        self.beta = nn.Parameter(torch.zeros(shape)) # the bias that is the new mean.
    def forward(self, X):
        #if we are in training mode, then we use the mean and variance of this batch.
        X_batch = self.batch_norm(X);
        X_instance = self.instance_norm(X);
        #X_batch = (X - mean) / torch.sqrt(var + self.epsilon); #epsilon is added for non-zero denominator
        #instance_mean = torch.mean(X, dim = (2,3), keepdim = True);
        #instance_var = torch.var(X, dim = (2,3), keepdim = True);
        #X_instance = (X - instance_mean) / torch.sqrt(instance_var + self.epsilon); #this is the instance value.
        X_transformed = self.rho * X_batch + (1 - self.rho) * X_instance;
        X_transformed = X_transformed * self.gamma + self.beta;
        return X_transformed;
class LayerNormalisation2d(nn.Module):
    def __init__(self, size = None):
        super(LayerNormalisation2d, self).__init__();
        self.epsilon = 1e-5; #it actually has no use for size, since it normalizes accross the channels as well.
        self.gamma = nn.Parameter(torch.ones((1, 1, 1, 1)));
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, 1)));
    def forward(self, X):
        mean = torch.mean(X, dim = (1,2,3), keepdim = True); #normalizes accross the channel dimension as well.
        var = torch.var(X, dim = (1,2,3), keepdim = True);
        X_transformed = (X - mean) / torch.sqrt(var + self.epsilon);
        X_transformed = X_transformed * self.gamma + self.beta;
        return X_transformed;
class GroupNormalisation2d(nn.Module):
    def __init__(self, size, groups = 8):
        super(GroupNormalisation2d, self).__init__();
        self.epsilon = 1e-5;
        self.gamma = nn.Parameter(torch.ones((1, size, 1, 1)));
        self.beta = nn.Parameter(torch.zeros((1, size, 1, 1)));
        self.groups = groups;
    def forward(self, X):
        shape = X.shape;
        X = X.view(shape[0], self.groups, shape[1]//self.groups, shape[2], shape[3]);
        mean = torch.mean(X, dim = (2,3,4), keepdim = True);
        var = torch.var(X, dim = (2,3,4), keepdim = True);
        X_transformed = (X - mean) / torch.sqrt(var + self.epsilon);
        X_transformed = X_transformed.view(shape);
        X_transformed = X_transformed * self.gamma + self.beta;
        return X_transformed;
class NoNormalisation(nn.Module):
    def __init__(self, size):
        super(NoNormalisation, self).__init__();
    def forward(self, X):
        return X; #no transformations applied.

# %% [markdown]
# ### Describing the ResNet class

# %%
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, norm = nn.BatchNorm2d):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1,bias=False)
        self.bn1 = norm(out_channels).to(device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,bias=False)
        self.bn2 = norm(out_channels).to(device=device)
        self.stride = stride;
        self.conv1x1 = None; self.bn1x1 = None; #Originally.
        if(self.stride != 1):
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0,bias=False)
            self.bn1x1 = nn.BatchNorm2d(out_channels,device=device);

    def forward(self, x):
#         residual = x;
        o = self.conv1(x)
        o = self.bn1(o);
        o = F.relu(o).to(device); #The first layer for the resnet block.
        o = self.conv2(o); 
        o = self.bn2(o); 
        if(self.stride != 1): #this means we have to perform 1x1 convolutions
            x = self.conv1x1(x); 
            x = self.bn1x1(x); #Applying the 1x1 convolutions to maintain the size.
        o += x; #inplace addition.
        o = F.relu(o); #the second layer output completed here.
        return o;
class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n, norm=nn.BatchNorm2d):
        super(ResNet, self).__init__();
        self.n = n;
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, device=device,bias=False);
        self.bn1 = norm(16).to(device=device); #of output size.
        self.relu = nn.ReLU();
        self.res16 = nn.ModuleList();
        for i in range(n):
            self.res16.append(ResNetBlock(16,16, norm=norm).to(device));
        self.res32 = nn.ModuleList();
        self.res32.append(ResNetBlock(16,32,2, norm=norm).to(device)); #1 Block which will change the size of the input.
        for i in range(n-1):
            self.res32.append(ResNetBlock(32,32,norm=norm).to(device));
        self.res64 = nn.ModuleList();
        self.res64.append(ResNetBlock(32,64,2,norm=norm).to(device));
        for i in range(n-1):
            self.res64.append(ResNetBlock(64,64,norm=norm).to(device));
        
        self.final_mean_pool = nn.AdaptiveAvgPool2d(output_size=(1,1));
        self.fc = nn.Linear(64, num_classes);

    def forward(self, o):
        o = self.conv1(o)
        o = self.bn1(o)
        o = self.relu(o)
        for i in range(len(self.res16)):
            o = self.res16[i](o)
        for i in range(len(self.res32)):
            o = self.res32[i](o);
        for i in range(len(self.res64)):
            o = self.res64[i](o);
        o = self.final_mean_pool(o); 
        o = o.view(o.size(0), -1);
#         o = torch.flatten(o, start_dim=1); #Flattening from after the batch index.
        o = self.fc(o); #final layer.
        return o;

# %%


# %% [markdown]
# ### creating the dataloaders

# %%
## Parameters for the network.32
num_classes = 25; 
n = 2; #6n + 2 layers.
in_channels = 3; #RGB images.
batch_size = 32; #Probably wont run on my laptop with just 4GB of VRAM.

# %%

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename);
    model.load_state_dict(checkpoint['model_state_dict']);
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
    epoch = checkpoint['epoch'];
    loss = checkpoint['loss'];
    return model, optimizer, epoch, loss;
def store_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            }, filename);

# %% [markdown]
# python3 infer.py -model_file <path to the trained model> --normalization [ bn | in | bin | ln | gn | nn | inbuilt ] --n [ 1 |  2 | 3  ] --test_data_file <path to the directory containing the images> --output_file <file containing the prediction in the same order as the images in directory>
# 
# Example: 

# %%
# model = ResNet(in_channels, num_classes, n, norm=BatchNorm2d).to(device);
#model = ResNet(in_channels, num_classes, n).to(device);
#print("Doing Inbuilt implementation with batch size 32");
# here we get the data from argparse, about what we want.
parser = argparse.ArgumentParser(description='ResNet implementation with different normalizations');
parser.add_argument('--normalization', type=str, default='inbuilt', help='normalization type for the network');
parser.add_argument('--n', type=int, default=2, help='n value for the network');
parser.add_argument('--test_data_file', type=str, default='', help='test data file');
parser.add_argument('--output_file', type=str, default='output.txt', help='output file');
parser.add_argument('--model_file', type=str, default='models/part_1.2_gn.pth', help='output file');
#how to use these args?

# %%
# args = parser.parse_args("--model_file models/part_1.2_gn.pth --normalization gn --n 2 --test_data_file birds_test/birds_test --output_file output.txt".split());
args = parser.parse_args();
norm = nn.BatchNorm2d; #by Default.
if(args.normalization == 'bn'):
    norm = BatchNorm2d;
elif(args.normalization == 'in'):
    norm = InstanceNormalisation2d;
elif(args.normalization == 'bin'):
    norm = BatchInstanceNormalisation2d;
elif(args.normalization == 'ln'):
    norm = LayerNormalisation2d;
elif(args.normalization == 'gn'):
    norm = GroupNormalisation2d;
elif(args.normalization == 'nn'):
    norm = NoNormalisation;

# %%
args._get_kwargs()

# %%
model = ResNet(in_channels, num_classes, args.n, norm=norm).to(device); #n should be equal to 2 hopefully.
optimizer = optim.Adam(model.parameters(), lr=0.001); #Adam optimizer, although it is not needed but load checkpoint uses it.
load_checkpoint(model, optimizer, args.model_file)
print( "DOING MY IMPLEMENTATION OF {} with batch_size".format(args.normalization), batch_size)

#Check accuracy on training and test to see how good our model is.
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# %%
class bird_dataset(Dataset):
    def __init__(self, datapath, train = True): #Either test, train, or val datafolder.
        self.datapath = datapath;
        self.train = train;
        if(train):
            folder_list = glob.glob(datapath + "/*");
            self.data = [];
            self.labels = set();
            for folder in folder_list:
                label = os.path.basename(folder); #gets the last name of the folder, which is the label.
                self.labels.add(label);
                file_list = glob.glob(folder + "/*");
                for file in file_list:
                    self.data.append((file, label));
            self.labels = list(self.labels);
            self.label_to_index = {label: i for i, label in enumerate(self.labels)};
        else:
            file_list = glob.glob(datapath + "/*");
            self.data = [];
            label = 'dunno';
            for file in file_list:
                self.data.append((file, label));
            self.labels = [label];
            self.label_to_index = {label: i for i, label in enumerate(self.labels)};
            #converts the label to an index.
    def __len__(self):
        return len(self.data);
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx];
        img = Image.open(img_path)
        img = np.array(img)/255; #normalizing the image, maybe not necessary but just to confirm.
        img = transforms.ToTensor()(img);
        label = self.label_to_index[label]; #using labels as indices for the classes, instead of names.
        #this label doesn't really correspond to anything in the dataset.
        # label_arr = np.zeros(len(self.labels));
        # label_arr[label] = 1;
        path = os.path.basename(img_path);
        return img, path;

# %%


# %%
batch_size = 4;
Test_loader = DataLoader(bird_dataset(args.test_data_file, train=False), batch_size=batch_size, shuffle=False); #lower batch size during testing.

# %%
class_to_idx = {'Asian-Green-Bee-Eater': 0, 'Brown-Headed-Barbet': 1, 'Cattle-Egret': 2, 'Common-Kingfisher': 3, 'Common-Myna': 4, 'Common-Rosefinch': 5, 'Common-Tailorbird': 6, 'Coppersmith-Barbet': 7, 'Forest-Wagtail': 8, 'Gray-Wagtail': 9, 'Hoopoe': 10, 'House-Crow': 11, 'Indian-Grey-Hornbill': 12, 'Indian-Peacock': 13, 'Indian-Pitta': 14, 'Indian-Roller': 15, 'Jungle-Babbler': 16, 'Northern-Lapwing': 17, 'Red-Wattled-Lapwing': 18, 'Ruddy-Shelduck': 19, 'Rufous-Treepie': 20, 'Sarus-Crane': 21, 'White-Breasted-Kingfisher': 22, 'White-Breasted-Waterhen': 23, 'White-Wagtail': 24}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# %%
outputs = {}; #outputs dictionary, for each line.
total_done = 0;
for x,y in Test_loader:
    indices = [i[:-4] for i in y];
    x = x.to(device, dtype = torch.float32);
    model_outputs = model(x);
    _, preds = model_outputs.max(1);
    for i in range(len(preds)):
        outputs[indices[i]] = preds[i];
    total_done += len(preds);
    if(total_done %(batch_size * 10) == 0):
        print("total done: ", total_done, end = "         \r");

# %%
#finally write the outputs to the file.
#first we make it a list of tuples and then order it by the indices
#then we write it to the file.
preds = [(key, outputs[key]) for key in outputs.keys()];
try:
    preds = sorted(preds, key = lambda x: int(x[0])); # Sort by the indices.
except Exception as e:
    print("couldnt sort by indices, maybe an error?", e);

print("beginning to write our predictions to {}".format(args.output_file));
with open(args.output_file, 'w') as f:
    for i in range(len(preds)):
        f.write(idx_to_class[preds[i][1].item()] + "\n");

# %%



