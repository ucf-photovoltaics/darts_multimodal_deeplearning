import os
import sys


import numpy as np
import time
import pandas as pd
import torch
import torchvision
from torchvision.io import read_image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from json import loads as str_to_dic
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import calc_classes_weight, exp_lr_scheduler, arrange_classes
import copy

#Settings
dataset_dir='/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/sxl2318/classification/dataset/UVF'
meta_data = '' #This will be added to the end of the save file name to distinguish the names 
class_setup_number = 3 # How do we want to reange the classses, 0 is default
use_gpu = True
BASE_LR = 0.0001
EPOCH_DECAY = 50 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.2 # factor by which the learning rate is reduced.
BATCH_SIZE = 100
MODEL_NAME = 'resnet18'
NUM_EPOCHS = 200
classes, num_classes  = arrange_classes (class_setup_number) 

class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    img_labels_rows = pd.read_csv(annotations_file)
    img_labels_rows = img_labels_rows.dropna() #dropping one with NAN
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{}']
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{"feature":"","number":""}']

    #creating the label vector
    self.img_label_vector = np.zeros((img_labels_rows.shape[0],num_classes),dtype=np.single)  #Vector to hold labels
    img_label_vector_temp =  np.zeros((1,num_classes),dtype=np.single) #used to create temporary vector for labels

    self.img_names = []  #list of all of the file names
    previous = "-1"
    j= -1
    invalid_flag = 1 #This flag is 1 when all the features found in an image belong to omitted classes (not by -1 in classes dictionary) 
    for i in range(img_labels_rows.shape[0]):

      if not (img_labels_rows.iloc[i, 0] ==  previous ):  #When ith row is a new file

        previous = img_labels_rows.iloc[i, 0]
        if invalid_flag == 0 : # It means that the previous image had at least one valid feature
          j = j +1
          self.img_names.append(img_labels_rows.iloc[i-1, 0])
          self.img_label_vector [j, :] = img_label_vector_temp 
          invalid_flag = 1

        img_label_vector_temp =  np.zeros((1,num_classes),dtype=np.single)

      label_string = img_labels_rows.iloc[i, 6]
      label = str_to_dic(label_string)['feature']
      label_num = classes[label]  #finding the index of label in classes array
      if not (label_num == -1):
        img_label_vector_temp [0, label_num] = 1
        invalid_flag = 0

    self.img_label_vector = self.img_label_vector[0:j+1,:]
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    
  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    try:
      img_path = os.path.join(self.img_dir, self.img_names[idx])
      image = read_image(img_path)
      label_vec = self.img_label_vector[idx, :]
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label_vec = self.target_transform(label_vec)
    except:
      print (os.path.join(self.img_dir, self.img_names[idx]))
      breakpoint()
    return image, label_vec


def train_model(model, criterion, optimizer, lr_scheduler, dataset_loaders,dset_sizes, num_epochs=100):
    since = time.time()

    best_model = model
    best_f1_score = -1
    f1_score_test = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        y_true = []
        y_pred = []


        # Each epoch has a training, test and validation phase
        for phase in ['train', 'test', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,BASE_LR,EPOCH_DECAY,DECAY_WEIGHT)
                model.train()  # Set model to training mode
            elif (phase == 'test'):
                model.eval()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #counter=0
            # Iterate over data.
            for data in dataset_loaders[phase]:

                inputs, labels = data
                # wrap them in Variable

                inputs, labels = inputs.cuda(), labels.cuda()
                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                sig = nn.Sigmoid()
                outputs = sig (model(inputs))
                preds = torch.floor(outputs.data+.5)

                loss = criterion(outputs, labels)

                y_true = y_true + labels.tolist()

                y_pred = y_pred + preds.tolist()
   
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # print evaluation statistics
                try:
                    running_loss += loss.item()
                    #running_corrects += torch.sum(preds == labels.data)
                except:
                    print('unexpected error')

            epoch_loss = running_loss / dset_sizes[phase]            
            f1_score_epoch = f1_score(np.array (y_true), np.array (y_pred), average='macro')

            if phase == 'test':
              f1_score_test = f1_score_epoch

            if phase == 'val':                 # We use the f1 score for validation set to choose the best model
              if f1_score_epoch > best_f1_score:
                best_f1_score = f1_score_epoch
                best_model = copy.deepcopy(model)
                print('new best accuracy for test = ', f1_score_test)
                print('new best accuracy for val = ', f1_score_epoch)

                    
            print('{} Loss: {:.4f} f1_score: {:.4f}'.format(
                phase, epoch_loss, f1_score_epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return best_model

def main():

  data_transforms = {
    'train': transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  
  training_data = CustomImageDataset(annotations_file= dataset_dir+ '/train/all_annotations_new.csv',
    img_dir = dataset_dir + '/train',transform = data_transforms['train'])

  test_data = CustomImageDataset(annotations_file= dataset_dir + '/test/all_annotations_new_test_corrected.csv',
    img_dir = dataset_dir+'/test',transform = data_transforms['test'])

  val_data = CustomImageDataset(annotations_file= dataset_dir + '/val/annotation_from_script.csv',
    img_dir = dataset_dir+'/val',transform = data_transforms['val'])
  
  train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True , num_workers=16)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False , num_workers=16)
  val_dataloader= DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False , num_workers=16)

  dataset_loaders = {
      'train':train_dataloader,
      'test' :test_dataloader,
      'val' : val_dataloader
  }


  dset_sizes = {
      'train':training_data.__len__(),
      'test' :test_data.__len__(),
      'val' : val_data.__len__()
  }

  # Finding wieght of each class for unblanced dataset
  class_weight, _ = calc_classes_weight (dataset_dir + '/train/all_annotations_new.csv', classes , num_classes)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  class_weight = torch.from_numpy(class_weight).float().to(device)

  if MODEL_NAME == 'densenet121':
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = 1024 #model_ft.fc.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'densenet161':
    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features #model_ft.fc.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'densenet169':
    model_ft = models.densenet169(pretrained=True)
    num_ftrs = model_ft.classifier.in_features #model_ft.fc.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'resnet18':
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'resnet50':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'wide_resnet50_2':
    model_ft = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'vgg11_bn':
    model_ft = models.vgg11_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'vgg13_bn':
    model_ft = models.vgg13_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'vgg16_bn':
    model_ft = models.vgg16_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)  
  elif MODEL_NAME == 'vit_b_16':
    model_ft = models.vit_b_16(pretrained=True)
    num_ftrs = model_ft.heads.head.in_features
    model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
  elif MODEL_NAME == 'vit_l_16':
    model_ft = models.vit_l_16(pretrained=True)
    num_ftrs = model_ft.heads.head.in_features
    model_ft.heads.head = nn.Linear(num_ftrs, num_classes)  



  #criterion = nn.CrossEntropyLoss(weight=class_weight)
  criterion = nn.BCELoss (weight=class_weight)
  if use_gpu:
    criterion.cuda()
    #model_ft.cuda()
    model_ft= nn.DataParallel(model_ft)
    model_ft.to(device)

  optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)



  # Run the functions and save the best model in the function model_ft.
  model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_loaders,dset_sizes,num_epochs=NUM_EPOCHS)

  # Save model
  torch.save(model_ft.module.state_dict(), 'fine_tuned_weights/' + MODEL_NAME + '_classes_' + str(class_setup_number) + '_' + meta_data  + '_' + 'weights.pt')
  #model_ft._save_to_state_dict(MODEL_NAME +  ' fine_tuned_best_model.pt')

if __name__ == "__main__":
  main()
  print ('All done')


# @title