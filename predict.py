
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
import json
from PIL import Image
import argparse
import time
start=time.time()
parser = argparse.ArgumentParser(                     #Creating a parser
                    prog='predict  the class of image ',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('image_dir', type=str,help='take dir of image')
parser.add_argument('checkpoint', type=str,help='the information of model ')
parser.add_argument('--top_k', type=int,required=False,default=1,help='the number of class to show')
parser.add_argument('--category_names', type=str,required=False,default='/content/drive/MyDrive/Image_Classifier_Project/cat_to_name.json',help='dir to jason file')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()  

image_dir=args.image_dir
checkpoint=args.checkpoint
top_k=args.top_k
category_names=args.category_names
gpu=args.gpu
print(args)
print(image_dir)
print(top_k)
print(category_names)
print(gpu)

if gpu :
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else :
  device =torch.device('cpu')
print(device)

checkpoint_new=torch.load(checkpoint)

model_name=checkpoint_new['model_name']
if model_name=='densenet121':
  model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
if model_name=='vgg13':
    model = models.vgg13(weights=models.DenseNet121_Weights.DEFAULT)

model.classifier = checkpoint_new['classifier']
model.load_state_dict(checkpoint_new['state_dict'])
model.class_to_idx = checkpoint_new['mapping']

def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # resize the images where the shortest side is 256 pixels
    size=256
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(size * (height / width))
    else:
        new_height = size
        new_width = int(size * (width / height))

    # Resize the image
    image = image.resize((new_width, new_height))

    # Crop  224x224
    left = (image.width - 224) / 2
    upper = (image.height - 224) / 2
    right = (image.width + 224) / 2
    lower = (image.height + 224) / 2
    im_crop = image.crop((left, upper, right, lower))

    # Convert PIL image to NumPy array
    np_image = np.array(image) / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    #  (color channel first)
    np_image = np_image.transpose((2, 0, 1)) #(height, width, channels) >>> (channels, height, width)

    tensor_image = torch.from_numpy(np_image).float()
    return tensor_image
   #return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))   #(channels, height, width)  >>>  ( height, width,channels)

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
image = Image.open(image_dir)
image = process_image(image)
imshow(image,)

plt.show()
def predict(image_path, model, topk=5 , device="cpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = process_image(image)
    model.to(device)
    model.eval()
    with torch.no_grad():
      logps=model.forward(image.unsqueeze(0))  #Adding the extra dimension at the beginning effectively makes it a batch of size 1, which is a common requirement for models that expect input in batch format

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}  #(class,idx) >>(idx,class)
    classes = [idx_to_class[idx] for idx in top_class.numpy()[0].tolist()] #tolist() converts the 1D NumPy array to a Python list
    top_p = [round(prob.item(), 5) for prob in top_p.numpy()[0]]  #convert top_p to list and round it
    return top_p, classes
    # TODO: Implement the code to predict the class from an image file

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
prop,classes=predict(image_dir,model,top_k,device)
classes = [cat_to_name[cls] for cls in classes]

for img_pr, img_class in zip(prop, classes):
    print(f'{img_class}: {img_pr:.2f}')

total_time=time.time()-start
print(f'the total time for prodict process is {total_time:0.4}')
if __name__ == "__main__":
  pass