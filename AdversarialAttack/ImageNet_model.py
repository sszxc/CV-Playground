# Author: Xuechao Zhang
# Date: May 18th, 2022
# Description: 
#   Use Net() and Data() to get ImageNet model and dataloader for FGSM attack

import torch
from torchvision import transforms
from PIL import Image
import re
import os
import sys
os.chdir(sys.path[0])  # 当前文件目录

def Net():
    """
    Load a pre-trained MobileNet V2 Model
    Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    """
    return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

def Data():
    """
    Build a Dataset with 1000 images, one per image-net class.
    Clone this ↓ repo and move on.
    Ref: https://github.com/EliSchwartz/imagenet-sample-images
    """
    imagenet_data = ImageNet_sample('imagenet-sample-images')  # 数据集路径
    test_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=1,
                                            shuffle=True)
    return test_loader    

class ImageNet_sample(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.all_image_paths = []  # 收集有效图片
        for x in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, x)
            if self.__is_image_file(x) and self.__is_RGB_file(path):  # 过滤灰度图、非图片文件
                self.all_image_paths.append(path)
                        
        # 下载 'index'-'class name' 对应文件
        if not os.path.exists("imagenet_classes.txt"):
            os.system("wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    def __is_image_file(self, filename):
        """
        判断文件是否是图片
        """
        return any(filename.endswith(extension) 
            for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def __is_RGB_file(self, filename):
        """
        判断文件是否是RGB图片
        """
        img = Image.open(filename)
        return img.mode == 'RGB'

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        # 图形预处理
        input_image = Image.open(self.all_image_paths[index])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化影响可视化效果 屏蔽
        ])
        img = preprocess(input_image)

        # 正则表达式找到文件名中的label, 到txt文件中找index
        file_name = re.search(r'n[0-9]{8}.+\.', self.all_image_paths[index]).group()
        class_name = file_name[10:-1].replace('_', ' ')
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        label = categories.index(class_name)

        return img, label

if __name__ == "__main__":
    net = Net()
    data = Data()
    print("stop here")