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
import pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

# %%


# %% [markdown]
# ### Describing the ResNet class

# %%
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.stride = stride;
        self.conv1x1 = None; self.bn1x1 = None; #Originally.
        if(self.stride != 1):
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, device=device)
            self.bn1x1 = nn.BatchNorm2d(out_channels,device=device);
    
    def forward(self, x):
        residual = x;
        o = self.conv1(x)
        o = self.bn1(o);
        o = self.relu(o).to(device); #The first layer for the resnet block.
        o = self.conv2(o); o = self.bn2(o); 
        if(self.stride != 1): #this means we have to perform 1x1 convolutions
            residual = self.conv1x1(residual); residual = self.bn1x1(residual); #Applying the 1x1 convolutions to maintain the size.
        o += residual; #inplace addition.
        o = self.relu(o); #the second layer output completed here.
        return o;


# %%
class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n):
        super(ResNet, self).__init__();
        self.n = n;
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, device=device);
        self.bn1 = nn.BatchNorm2d(16); #of output size.
        self.relu = nn.ReLU();
        self.res16 = [];
        for i in range(n):
            self.res16.append(ResNetBlock(16,16).to(device));
        self.res32 = [ResNetBlock(16,32,2).to(device)]; #1 Block which will change the size of the input.
        for i in range(n-1):
            self.res32.append(ResNetBlock(32,32).to(device));
        self.res64 = [ResNetBlock(32,64,2).to(device)];
        for i in range(n-1):
            self.res64.append(ResNetBlock(64,64).to(device));
        self.final_mean_pool = nn.AvgPool2d(kernel_size=64, stride=1);
        self.fc = nn.Linear(64, num_classes);

    def forward(self, x):
        x = x.to(device); #incase it is not.
        o = self.conv1(x).to(device); o = self.bn1(o).to(device); o = self.relu(o).to(device);
        for i in range(self.n):
            o = self.res16[i](o).to(device);
        for i in range(self.n):
            o = self.res32[i](o);
        for i in range(self.n):
            o = self.res64[i](o);
        o = self.final_mean_pool(o); 
        # o = o.view(o.size(0), -1);
        o = torch.flatten(o, start_dim=1); #Flattening from after the batch index.
        o = self.fc(o); #final layer.
        return o;
        

# %%
class ResNet2(nn.Module):
    def __init__(self, in_channels, num_classes, n):
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, device=device); # 16 filters, kernel size 3x3. output shape is same as input shape due to padding = 1
        self.n = n;
        #Now we have the next 3 blocks of ResNet. 
        # The first n blocks have 2n filters of size 3x3, and 16 filters, with residual connection between each 2 consecutive filters.
        self.res16 = [];
        for i in range(2*n): #2n channels of 16 filters each. Need to have residual connections between them too.
            self.res16.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, device= device)); 

        self.res16_32_1x1 = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0, device=device); #1x1 convolution to increase the number of filters to 32, and halve the feature map size from 256x256 to 128x128.
        self.res32 = [nn.Conv2d(16,32, kernel_size=3, stride=2, padding=1, device=device)]; #Halves the feature map size from 256x256 to 128x128, while increasing filters to 32
        for i in range(2*n-1):
            self.res32.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, device=device));

        self.res32_64_1x1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, device=device); #1x1 convolution to increase the number of filters to 64, and halve the feature map size from 128x128 to 64x64.
        self.res64 = [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, device=device)]; #Halves the feature map size from 128x128 to 64x64, while increasing filters to 64
        for i in range(2*n-1):
            self.res64.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, device=device));

        self.final_mean_pool = nn.AvgPool2d(kernel_size=64, stride=1); #Average pooling to get the mean of the 64x64 feature map
        self.fc = nn.Linear(64, num_classes, device=device); #Fully connected layer to output the class scores

    def forward(self, x):
        x = F.relu(self.conv1(x)); #First Convolutional layer.
        #Now we have the next 3 blocks of ResNet.
        #For the first n blocks. 
        for i in range(0,2*self.n-1,2): #with a step of 2, so we pass over each residual connection.
            x = F.relu(self.res16[i+1](F.relu(self.res16[i](x))) + x); #The output is complete here.
        # return x;
        #Now we have a residual of shape 16x256x256, we need to pass it through to the next layer of 32x128x128.
        res = self.res16_32_1x1(x); #to match the dimensions of the next layer.
        x = F.relu(self.res32[1](F.relu(self.res32[0](x))) + res);
        for i in range(2,2*self.n-1,2):
            x = F.relu(self.res32[i+1](F.relu(self.res32[i](x))) + x);

        res = self.res32_64_1x1(x);
        x = F.relu(self.res64[1](F.relu(self.res64[0](x))) + res);
        for i in range(2,2*self.n-1,2):
            x = F.relu(self.res64[i+1](F.relu(self.res64[i](x))) + x);
        # print("out_shape: ", x.shape); #The output shape is 64x64x64
        x = self.final_mean_pool(x); #Average pooling to get the mean of the 64x64 feature map
        # print("after avgpool:" ,x.shape);
        x = torch.flatten(x, start_dim=1); #Flatten the output to pass it to the fully connected layer. The first dimension is the batch.
        x = self.fc(x); #Fully connected layer to output the class scores
        return x;




        

# %%
class bird_dataset(Dataset):
    def __init__(self, datapath): #Either test, train, or val datafolder.
        self.datapath = datapath;
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
    
    def __len__(self):
        return len(self.data);
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx];
        img = read_image(img_path)
        img = img/255;
        # print(img);
        # img = transforms.ToTensor()(img); #converts the image to a tensor, but read_image already does this.
        label = self.label_to_index[label]; #using labels as indices for the classes, instead of names.
        # label_arr = np.zeros(len(self.labels));
        # label_arr[label] = 1;
        return img, label;



# %% [markdown]
# ### creating the dataloaders

# %%
## Parameters for the network.32
num_classes = 25; 
n = 2; #6n + 2 layers.
in_channels = 3; #RGB images.
batch_size = 32; #Probably wont run on my laptop with just 4GB of VRAM.
initial_learning_rate = 0.01;
num_epochs = 50; 

# %%
Train_loader = DataLoader(bird_dataset("Birds_25\\train"), batch_size=batch_size, shuffle=True); #This is how to use the DataLoader to get batches of data.
Test_loader = DataLoader(bird_dataset("Birds_25\\test"), batch_size=batch_size, shuffle=True);
Val_loader = DataLoader(bird_dataset("Birds_25\\val"), batch_size=batch_size, shuffle=True);

# %%
model = ResNet(in_channels, num_classes, n).to(device);

# %%
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr= initial_learning_rate);
optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate);

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

# %%

def train_model(model):
    for epoch in range(num_epochs):
        start = time.time();
        mean_loss = 0; total_batches = 0;
        print("epoch: ", epoch+1);
        for i, (images, labels) in enumerate(Train_loader):
            total_batches += 1;
            images = images.to(device);
            labels = labels.to(device);
            #Forward pass
            outputs = model(images);
            loss = criterion(outputs, labels);
            #Backward pass
            optimizer.zero_grad(); #Zeroes the gradients before backpropagation.
            loss.backward(); #Backpropagation.
            optimizer.step(); #Updates the weights.
            mean_loss += loss.item();
            if(i%25 == 0):
                store_checkpoint(model, optimizer, epoch, loss.item(), "checkpoint.pth");
            print("batch: ", i+1, "loss: ", loss.item(), end = "          \r");
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] - 0.0002; #Decay the learning rate by a constant after each epoch.
        end = time.time();
        store_checkpoint(model, optimizer, epoch, loss.item(), "checkpoint" + str(epoch) + ".pth");
        print(epoch, "th epoch: ", end-start, " seconds,   ", "mean loss: ", mean_loss/total_batches, "          ");

# %%
optimizer.param_groups

# %%
# load_checkpoint(model, optimizer, "checkpoint.pth");

# %%
train_model(model);

# %%
#Check accuracy on training and test to see how good our model is.
def check_accuracy(loader, model):
    correct = 0; num_samples = 0;
    model.eval(); #Sets it into evaluation mode, so no dropout or batchnorm
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device);
            y = y.to(device);
            #x = x.reshape(x.shape[0], -1);
            scores = model(x);
            _, predictions = scores.max(1);
            correct += (predictions == y).sum();
            num_samples += predictions.size(0);
            print("partial: ", correct, "/", num_samples, " = ", correct/num_samples, "%", end = "          \r");

        print(f"Got {correct} / {num_samples} with accuracy {float(correct)/float(num_samples)*100:.2f}");
    model.train();

# %%


# %%
# load_checkpoint(model, optimizer, "checkpoint.pth");
check_accuracy(Val_loader, model);
check_accuracy(Test_loader, model);

# %%
# load_checkpoint(model, optimizer, "checkpoint_kaggle.pth");
check_accuracy(Val_loader, model);
check_accuracy(Test_loader, model);

# %%
store_checkpoint(model, optimizer, 51, 2, "chkpt.pth");

# %%



