
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from dataset import ComposeWithMultiTensor,RandomApplyWithMultiTensor,RandomResizedCropWithMultiTensor,RandomHorizontalFlipWithMultiTensor,RandomVerticalFlipWithMultiTensor,RandomGrayscaleWithMultiTensor

basic_transform = transforms.Compose(
[
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_list,
                 inverse = False,
                 mask_threshold= -1,
                 use_custom_transform=False): 
        inpaint_dir_list = []
        mask_dir_list = []
        origin_dir_list = []
        for data_dir in data_dir_list:
            name_list = os.listdir(data_dir+'/inpaint')
            for name in name_list:
                inpaint_dir_list.append(data_dir + "inpaint/" + name.strip(".jpg")+".jpg")
                mask_dir_list.append(data_dir + "mask/" + name.strip(".jpg")+".jpg.npy")
                origin_dir_list.append(data_dir + "origin/" + name.strip(".jpg")+".jpg")
        
        self.inpaint_dir_list = inpaint_dir_list
        self.mask_dir_list = mask_dir_list
        self.origin_dir_list = origin_dir_list
        self.mask_threshold = mask_threshold
        self.inverse = inverse
        self.use_custom_transform = use_custom_transform
        self.basic_transform = basic_transform
    def __len__(self): 
        return len(self.inpaint_dir_list)

    def __getitem__(self, idx):
        origin = Image.open(self.origin_dir_list[idx])
        inpaint = Image.open(self.inpaint_dir_list[idx])
        mask = np.load(self.mask_dir_list[idx])

        if self.inverse:
            mask = 1. - mask  # 原本的code 是 0 為mask 區域, Platte 則是 1 為mask區域
        if self.mask_threshold != -1:
            mask[mask >= self.mask_threshold] = 1
            mask[mask <  self.mask_threshold] = 0


        mask = torch.from_numpy(mask).permute(2,0,1).to(dtype = torch.float32)

        origin = self.basic_transform(origin)
        inpaint = self.basic_transform(inpaint)

        if self.use_custom_transform:
            transforms_f = ComposeWithMultiTensor([
                RandomGrayscaleWithMultiTensor(p=0.5),
                RandomVerticalFlipWithMultiTensor(p=0.5),
                RandomHorizontalFlipWithMultiTensor(p=0.5),
                RandomApplyWithMultiTensor([RandomResizedCropWithMultiTensor(size = (origin.shape[-2],origin.shape[-1]))], p=0.5)
            ])

            origin, warpped, mask = transforms_f([origin, inpaint, mask])
            
            mask = torch.clamp(mask, min=0, max=1)

    
        return origin, inpaint, mask

if __name__ == "__main__":
    path = "../../Inpaint_dataset/"
    train_set = InpaintDataset(data_dir_list = [path], use_custom_transform = True)
    origin, inpaint, mask = train_set[11]
    print(origin.shape)
    print(inpaint.shape)
    print(mask.shape)
    toimage = transforms.ToPILImage()
    ori = toimage(origin)
    inp = toimage(inpaint)
    mas = toimage(mask)
    ori.save("ori.jpg")
    inp.save("inp.jpg")
    mas.save("mas.jpg")