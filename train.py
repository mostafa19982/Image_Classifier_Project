import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
import json
import time
from PIL import Image
import argparse
parser = argparse.ArgumentParser(                     #Creating a parser
                    prog='trainning model',
                    description='What the program does',
                    epilog='Text at the bottom of help')
# parser.print_help()
parser.add_argument('data_dir', type=str,help='take dir of data')
parser.add_argument('--save_dir', type=str,required=False,default='/content/drive/MyDrive/Image_Classifier_Project/checkpoint2.pth',help='dir to save')
parser.add_argument('--arch', type=str,required=False,default='densenet121',choices=['densenet121','vgg13'],help='the model name is dendenet or vgg13 ')
parser.add_argument('--learning_rate',type=int,required=False,default=0.003,help='the learning rate of trainning neural network')
parser.add_argument('--hidden_units', required=False,type=int)
parser.add_argument('--epochs',type=int,required=False,default=5)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()                                           #25088 vgg
print(args.data_dir)
print(args.save_dir)
print(args.arch)
print(args.learning_rate)
print(args.hidden_units)
print(args.epochs)
print(args.gpu)

data_dir=args.data_dir
save_dir=args.save_dir
arch=args.arch
learning_rate=args.learning_rate
hidden_units=args.hidden_units
epochs=args.epochs
gpu=args.gpu

if gpu :
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else :
  device =torch.device('cpu')
print(device)


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
valid_transforms=test_transforms

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_data = datasets.ImageFolder(train_dir , transform=train_transforms)
test_data = datasets.ImageFolder(test_dir , transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir , transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

if hidden_units ==None:
  if arch =='vgg13':
    hidden_units=12000
  if arch=='densenet121':
    hidden_units=512

if arch =='vgg13':
  fc1=25088
  model=models.vgg13(weights=models.VGG13_Weights.DEFAULT)
if arch=='densenet121':
  fc1=1024
  model=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

for param in model.parameters():
        param.requires_grad = False 

classifier=nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(fc1,hidden_units)),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(hidden_units,102)),
    ('output',nn.LogSoftmax(dim=1))

]))

model.classifier=classifier
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=0.003)
model.to(device)
#model=torch.load('/content/drive/MyDrive/my_second_project/train0,903.pth')
step=0
print_every=5
start_train=time.time()
print(f'start training use {arch} arch')
for epoch in range(epochs):
  start_epoch=time.time()
  running_loss = 0
  for input,label in trainloader:
    input , label = input.to(device) , label.to(device)
    step += 1
    logps = model.forward(input)
    loss = criterion(logps,label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss+=loss.item()

    if step % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader)*100:.3f}%")
            running_loss = 0
            model.train()
  epoch_time = time.time() - start_epoch
  print(f'the {epoch+1} epoch take a {int(epoch_time//60)}:{int(epoch_time%60)}')
total_time = time.time() - start_train
print(f'the total training time is {int(total_time//60)}:{int(total_time%60)}')


model.to('cpu')
model.class_to_idx = train_data.class_to_idx
checkpoint={'classifier':model.classifier,
             'state_dict':model.state_dict(),
             'mapping':model.class_to_idx,
             'model_name':arch,
             'epochs':epochs,
             'lr': learning_rate,
             }
torch.save(checkpoint,save_dir)

if __name__ == "__main__":
  pass
