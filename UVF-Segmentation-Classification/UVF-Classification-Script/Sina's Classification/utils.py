import numpy as np
import pandas as pd
from json import loads as str_to_dic

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch, weight_dacay):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (weight_dacay**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


#counting how many instances of each class in the training dataset
def calc_classes_weight (filepath, classes_dict, num_classes):

  count = np.zeros(num_classes)
  img_labels = pd.read_csv(filepath)
  img_labels = img_labels.dropna() #dropping one with NAN
  img_labels = img_labels[img_labels.region_attributes != '{}']
  img_labels = img_labels[img_labels.region_attributes != '{"feature":"","number":""}'] 

  for idx in range(0,img_labels.shape[0]):
    label_string = img_labels.iloc[idx, 6]
    label = str_to_dic(label_string)['feature']
    label_num = classes_dict[label]
    count[label_num] = count[label_num] + 1

  total_samples = np.sum(count)

  return 1/(count/total_samples), count

# return classes based on the input number
def arrange_classes (class_setup_number):
  if class_setup_number == 0:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':3, 'hotspot':4, 'finger_corrosion':5, 'near_busbar':6,
           'misc':7, 'busbar_crack':8, 'busbar_crrack':8 , 'busbbar_crack':8, 'shattered':9,'shattered ':9}
    num_classes = 10

  elif  class_setup_number == 1:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':2, 'hotspot':3, 'finger_corrosion':3, 'near_busbar':2,
           'misc':3, 'busbar_crack':2, 'busbar_crrack':2 , 'busbbar_crack':2, 'shattered':2,'shattered ':2}
    num_classes = 4

  elif  class_setup_number == 2:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':4, 'hotspot':-1, 'finger_corrosion':3, 'near_busbar':4,
           'misc':-1, 'busbar_crack':3, 'busbar_crrack':3 , 'busbbar_crack':3, 'shattered':2,'shattered ':2}
    num_classes = 5

  elif  class_setup_number == 3:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':3, 'hotspot':4, 'finger_corrosion':5, 'near_busbar':6,
           'misc':-1, 'busbar_crack':7, 'busbar_crrack':7 , 'busbbar_crack':7, 'shattered':8,'shattered ':8}
    num_classes = 9

  elif  class_setup_number == 4:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':2, 'hotspot':3, 'finger_corrosion':3, 'near_busbar':2,
           'misc':-1, 'busbar_crack':2, 'busbar_crrack':2 , 'busbbar_crack':2, 'shattered':2,'shattered ':2}
    num_classes = 4

  elif  class_setup_number == 5:
    classes = {'Square':0, 'Square ':0, 'square':0, 'ring':1, 'crack':2, 'cracxk':2, 'bright_crack':4, 'hotspot':5, 'finger_corrosion':3, 'near_busbar':4,
           'misc':-1, 'busbar_crack':3, 'busbar_crrack':3 , 'busbbar_crack':3, 'shattered':2,'shattered ':2}
    num_classes = 6

  else:
    raise ("Unknow classes Setup")
  return classes, num_classes

