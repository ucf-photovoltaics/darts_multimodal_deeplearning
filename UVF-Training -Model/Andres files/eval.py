import os

dataset_dir='/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/sxl2318/classification/dataset/UVF'
failure_dir = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/sxl2318/classification/failure_cases'
import numpy as np
import time
import pandas as pd
import torch
import torchvision  
from torchvision.io import read_image, write_jpeg
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from json import loads as str_to_dic
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import f1_score, confusion_matrix,  multilabel_confusion_matrix, classification_report
from utils import calc_classes_weight, exp_lr_scheduler, arrange_classes

use_gpu = True
meta_data = '' #This will be added to the end of the save file name to distinguish the names 
class_setup_number = 3 # How do we want to reange the classses, 0 is default
#NUM_CLASSES = 10
BATCH_SIZE = 32
MODEL_NAME = 'resnet18'
#classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':3, 'hotspot':4, 'finger_corrosion':5, 'near_busbar':6,
#           'misc':7, 'busbar_crack':8, 'busbar_crrack':8 , 'busbbar_crack':8, 'shattered':9,'shattered ':9}
classes, num_classes  = arrange_classes (class_setup_number) 

class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    img_labels_rows = pd.read_csv(annotations_file)
    img_labels_rows = img_labels_rows.dropna() #dropping one with NAN
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{}']
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{"feature":"","number":""}']

    #creating the label vector
    self.img_label_vector = np.zeros((img_labels_rows.shape[0],num_classes),dtype=np.single)  #Vector to hold labels
    self.img_names = []  #list of all of the file names
    previous = "-1"
    j=-1
    for i in range(img_labels_rows.shape[0]):

      if not (img_labels_rows.iloc[i, 0] ==  previous ):  #When ith row is a new file
        j = j +1
        previous = img_labels_rows.iloc[i, 0]
        self.img_names.append(img_labels_rows.iloc[i, 0])


      label_string = img_labels_rows.iloc[i, 6]
      label = str_to_dic(label_string)['feature']
      label_num = classes[label]  #finding the index of label in classes array

      self.img_label_vector[j, label_num] = 1


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

def test_model(model, criterion, dataset_loaders,dset_sizes,failure_analysis= None):
    since = time.time()

    best_model = model
    best_acc = 0.0

    
    y_true = []
    y_pred = []

    model.eval()
    mode='test'

    running_loss = 0.0
    running_corrects = 0

    phase = 'test'
    failure_counter = 0

    for i, data in enumerate (dataset_loaders['test']):

      inputs, labels = data
      # wrap them in Variable

      inputs, labels = inputs.cuda(), labels.cuda()
      sig = nn.Sigmoid()
      outputs = sig (model(inputs))
      preds = torch.floor(outputs.data+.5)
      loss = criterion(outputs, labels)

      y_true = y_true + labels.tolist()
      y_pred = y_pred + preds.tolist()
   
      running_loss += loss.item()
      #running_corrects += torch.sum(preds == labels.data)

      if failure_analysis is not None:
        failure_name_list = []
        for i, (label,pred) in enumerate(zip (labels.tolist(),preds.tolist())):
          if label == failure_analysis [0] and pred == failure_analysis [1]:
            failure_counter = failure_counter +1             
            sample  = img_name[i]
            #breakpoint()
            failure_name_list.append (sample)

    epoch_loss = running_loss / dset_sizes[phase]
    #epoch_acc = running_corrects.item() / float(dset_sizes[phase])

    if failure_analysis is not None:
      with open(failure_dir+'/failure.txt', 'w') as f:
        for fail_sample in failure_name_list:
          f.writelines(fail_sample + "\n")
          #breakpoint()


    #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #phase, epoch_loss, epoch_acc))


            
    #print ("f1_score is ")
    #print (f1_score(y_true, y_pred, average='macro'))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print ('images processes per second = ' , dset_sizes[phase] / time_elapsed)
    metrics = {'f1_score_macro':  f1_score(np.array (y_true), np.array (y_pred), average='macro') , 
      'f1_score': f1_score(np.array (y_true), np.array (y_pred), average=None), 
      'f1_score_micro': f1_score(np.array (y_true), np.array (y_pred), average='micro'),
      'confusion_matrix' : multilabel_confusion_matrix ( np.array (y_true) ,  np.array  (y_pred )), #,normalize = 'true')}
      'report' : classification_report( np.array (y_true) ,  np.array  (y_pred ))}
    return metrics

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
  }

  
  training_data = CustomImageDataset(annotations_file= dataset_dir+ '/train/all_annotations_new.csv',
    img_dir = dataset_dir + '/train',transform = data_transforms['train'])

  test_data = CustomImageDataset(annotations_file= dataset_dir + '/test/all_annotations_new_test_corrected.csv',
    img_dir = dataset_dir+'/test',transform = data_transforms['test'])

  
  train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
  dataset_loaders = {
      'train':train_dataloader,
      'test' :test_dataloader
  }


  dset_sizes = {
      'train':training_data.__len__(),
      'test' :test_data.__len__()
  }

  # Finding wieght of each class for unblanced dataset (Train)
  class_weight , numbers_train = calc_classes_weight (dataset_dir + '/train/all_annotations_new.csv', classes, num_classes)
  _ , numbers_test = calc_classes_weight (dataset_dir + '/test/all_annotations_new_test_corrected.csv', classes, num_classes)

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

    if use_gpu:
      model_ft= nn.DataParallel(model_ft)
      model_ft.to(device) 


  try:
    print ('loading weights from ' + 'fine_tuned_weights/' + MODEL_NAME + '_classes_' + str(class_setup_number) + '_' + meta_data  + '_' + 'weights.pt')
    #torch.load('fine_tuned_weights/' + MODEL_NAME + 'weights.pt', model_ft.state_dict())
    model_ft.load_state_dict(torch.load('fine_tuned_weights/' + MODEL_NAME + '_classes_' + str(class_setup_number) + '_' + meta_data  + '_' + 'weights.pt'))
  except:
    print ('Unable to load weights')
    breakpoint()

  criterion = nn.CrossEntropyLoss(weight=class_weight)

  if use_gpu:
    criterion.cuda()
    model_ft.cuda()

  # Run the functions and load the model 
  metrics = test_model(model_ft, criterion, dataset_loaders, dset_sizes ,failure_analysis = (2,1))
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print (metrics['confusion_matrix'])
  print (metrics['f1_score_macro'])
  print (metrics['f1_score_micro'])
  print (metrics['f1_score'])
  print (metrics['report'])
  print(' Number of parameters = ', sum(p.numel() for p in model_ft.parameters()))


if __name__ == "__main__":
  main()
