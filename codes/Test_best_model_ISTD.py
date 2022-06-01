import os,cv2
import torch,math,random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import matplotlib.image as img
from dataset import my_dataset,my_dataset_threeIn_test
import torchvision.transforms as transforms
from IterModel3 import Deshadow_netS4
from UTILS import compute_psnr

trans_eval = transforms.Compose(
        [transforms.Resize([256,256]),
         transforms.ToTensor()
        ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_PATH = '/ghome/zhuyr/AAAI22_shadow_removal/ISTD.pth'
test_results_path = '/gdata1/zhuyr/Derain_torch/AAAAA_Results_ISTD/'
if not os.path.exists(test_results_path):
    os.mkdir(test_results_path)

val_in_path = "/gdata1/zhuyr/AAAI22_shadow_removal/shadow_ISTD_test/test_A/"
val_gt_path = "/gdata1/zhuyr/AAAI22_shadow_removal/shadow_ISTD_test/test_C/"
val_mask_path = "/gdata1/zhuyr/AAAI22_shadow_removal/shadow_ISTD_test/test_B/"
filenames_val_in = os.listdir(val_in_path)
filenames_val_gt = os.listdir(val_gt_path)
filenames_val_mask = os.listdir(val_mask_path)
print("dataset :",filenames_val_mask==filenames_val_in)
print("dataset:",filenames_val_in==filenames_val_gt)
print("-"*100)

if __name__ == '__main__':
    net =Deshadow_netS4(ex1=6,ex2=4).to(device)
    print('#generator parameters:',sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load(SAVE_PATH))
    eval_data = my_dataset_threeIn_test(
        root_in=val_in_path,root_label =val_gt_path
       ,root_mask=val_mask_path,transform=trans_eval)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1,num_workers=1)
    net.eval()
    with torch.no_grad():
        psnr = 0
        rmse = 0
        shadowfree_rmse =0
        shadow_rmse =0
        total_psnr = 0.0
        for i, (data_in, label,mask,name) in enumerate(eval_loader, 0):
            print('name',name,name[0],'-'*30,i)
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            masks = Variable(mask).to(device)
            _, _, _, out_eval = net(inputs, masks)
            out_eval = torch.clamp(out_eval, 0., 1.)
            total_psnr += compute_psnr(out_eval,labels)
            out_eval_np = np.squeeze(out_eval.cpu().numpy())
            out_eval_np_ = out_eval_np.transpose((1,2,0))
            img.imsave(test_results_path + name[0],np.uint8(out_eval_np_ * 255.))
        print('Average psnr on ISTD dataset',total_psnr/len(eval_loader))

