import torch,os
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class my_dataset(Dataset):
    def __init__(self,root_in,root_label,transform =None):
        super(my_dataset,self).__init__()
        #in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)

        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
        else:
            data_IN =np.asarray(in_img)
            data_IN = torch.from_numpy(data_IN)
            data_GT = np.asarray(gt_img)
            data_GT = torch.from_numpy(data_GT)
        return data_IN,data_GT
    def __len__(self):
        return len(self.imgs_in)

class my_dataset_threeIn(Dataset):
    def __init__(self,root_in,root_label,root_mask,transform =None):
        super(my_dataset_threeIn,self).__init__()
        #in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        #mask_imgs
        mask_files = os.listdir(root_mask)
        self.imgs_mask = [os.path.join(root_mask, k) for k in mask_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        mask_img_path = self.imgs_mask[index]
        mask_img = Image.open(mask_img_path)
        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
            data_MASK = self.transform(mask_img)
        else:
            data_IN =np.asarray(in_img)
            data_IN = torch.from_numpy(data_IN)
            data_GT = np.asarray(gt_img)
            data_GT = torch.from_numpy(data_GT)
            data_MASK = np.asarray(mask_img)
            data_MASK = torch.from_numpy(data_MASK)
        return data_IN,data_GT,data_MASK
    def __len__(self):
        return len(self.imgs_in)


class my_dataset_threeIn_test(Dataset):
    def __init__(self,root_in,root_label,root_mask,transform =None):
        super(my_dataset_threeIn_test,self).__init__()
        #in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        #mask_imgs
        mask_files = os.listdir(root_mask)
        self.imgs_mask = [os.path.join(root_mask, k) for k in mask_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        mask_img_path = self.imgs_mask[index]
        mask_img = Image.open(mask_img_path)
        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
            data_MASK = self.transform(mask_img)
        else:
            data_IN =np.asarray(in_img)
            data_IN = torch.from_numpy(data_IN)
            data_GT = np.asarray(gt_img)
            data_GT = torch.from_numpy(data_GT)
            data_MASK = np.asarray(mask_img)
            data_MASK = torch.from_numpy(data_MASK)
        return data_IN,data_GT,data_MASK,img_name
    def __len__(self):
        return len(self.imgs_in)

class my_datasetY(Dataset):
    def __init__(self,root_in,root_label,transform =None):
        super(my_datasetY,self).__init__()
        #in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        #print("gt_img.shape",gt_img.size)

        if self.transform:
            in_img = np.expand_dims(np.asarray(in_img),-1)
            data_IN = self.transform(in_img)
            gt_img = np.expand_dims(np.asarray(gt_img), -1)
            data_GT = self.transform(gt_img)
        else:
            data_IN =np.expand_dims(np.asarray(in_img)/255.0,0)
            data_IN = torch.as_tensor(data_IN,torch.float32)
            #print("data_IN.shape", data_IN.size())
            data_GT = np.expand_dims(np.asarray(gt_img)/255.0)
            data_GT = torch.as_tensor(data_GT,torch.float32)
        return data_IN,data_GT
    def __len__(self):
        return len(self.imgs_in)