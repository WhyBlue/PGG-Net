import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from osgeo import gdal
from models.PGGNet.geo import ComputeSlope,ComputeAspect,hillshade

dataset_dir = "Dataset"

def read_txt_3(path):
    backs, bathys, seafloors, labels = [], [], [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            back, bathy, seafloor, label = line.strip().split()
            backs.append(back.replace('\\','/'))
            bathys.append(bathy.replace('\\','/'))
            seafloors.append(seafloor.replace('\\','/'))
            labels.append(label.replace('\\','/'))
    return backs, bathys, seafloors, labels

def read_label(filename):
    dataset=gdal.Open(filename)    #打开文件
 
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数
 
    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset 
    return im_data

def read_seafloor(seafloor_path):
    dataset_seafloor = gdal.Open(seafloor_path)  # 打开文件
    im_width_seafloor = dataset_seafloor.RasterXSize  # 栅格矩阵的列数
    im_height_seafloor = dataset_seafloor.RasterYSize  # 栅格矩阵的行数
    im_data_seafloor = dataset_seafloor.ReadAsArray(0, 0, im_width_seafloor, im_height_seafloor)  # 将数据写成数组，对应栅格矩阵
    # im_data_seafloor = np.expand_dims(im_data_seafloor,0)
    # arr = np.zeros((3, im_height_seafloor, im_width_seafloor))
    # for i in range(im_height_seafloor):
    #     for j in range(im_width_seafloor):
    #         arr[im_data_seafloor[i][j]-1][i][j] = 1

    im_data_seafloor[np.where(im_data_seafloor==0)] = 1
    im_data_seafloor = im_data_seafloor - 1

    return im_data_seafloor

def read_bathy(bathy_path):
    dataset_bathy = gdal.Open(bathy_path)  # 打开文件
    im_width_bathy = dataset_bathy.RasterXSize  # 栅格矩阵的列数
    im_height_bathy = dataset_bathy.RasterYSize  # 栅格矩阵的行数
    im_data_bathy = dataset_bathy.ReadAsArray(0, 0, im_width_bathy, im_height_bathy).astype('float64')  # 将数据写成数组，对应栅格矩阵
    # for i in range(im_height_bathy):
    #     for j in range(im_width_bathy):
    #         if im_data_bathy[i][j] < -86.98:
    #             im_data_bathy[i][j] = -86.98
    im_data_bathy = np.expand_dims(im_data_bathy, 0)

    return im_data_bathy.astype('float64')

def read_tiff_3(back_path, bathy_path, is_train):
    dataset_back = gdal.Open(back_path)  # 打开文件

    im_width_back = dataset_back.RasterXSize  # 栅格矩阵的列数
    im_height_back = dataset_back.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data_back = dataset_back.ReadAsArray(0, 0, im_width_back, im_height_back).astype('float64')  # 将数据写成数组，对应栅格矩阵

    im_data_back = np.expand_dims(im_data_back,0)

    im_data_bathy = read_bathy(bathy_path)

    # im_data_seafloor = read_seafloor(seafloor_path)

    im_data = np.concatenate([im_data_back, im_data_bathy], axis=0)

    max = [254, -117.12]
    for i in range(len(im_data)):
        im_data[i, :, :] = im_data[i, :, :] / max[i]

    del dataset_back
    return im_data

def priori_knowledge(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype('float64')  # 将数据写成数组，对应栅格矩阵

    Slope = np.expand_dims(ComputeSlope(im_data,x_cellsize=2,y_cellsize=2), 0)

    Aspect = np.expand_dims(ComputeAspect(im_data,x_cellsize=2,y_cellsize=2), 0)
    hs = np.expand_dims(hillshade(im_data), 0)

    out = np.concatenate([Slope,Aspect,hs], axis=0)

    return out

class UnetDataset(Dataset):
    def __init__(self, txtpath, transform, train=True):
        super().__init__()
        # self.ims, self.labels = read_txt(txtpath)
        self.backs, self.bathys, self.seafloors, self.labels = read_txt_3(txtpath)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        root_dir = dataset_dir
        back_path = os.path.join(root_dir,self.backs[index])
        bathy_path = os.path.join(root_dir,self.bathys[index])
        seafloor_path = os.path.join(root_dir,self.seafloors[index])
        # im_path = os.path.join(root_dir,self.ims[index])
        label_path = os.path.join(root_dir,self.labels[index])
        if_train = self.train

        image = read_tiff_3(back_path, bathy_path, if_train)
        # image = np.resize(image, (2,224,224))
        image = np.array(image)
        image = np.transpose(image,(1,2,0))
        image = transforms.ToTensor()(image)
        image = image.to(torch.float32).cuda()
        image = self.transform(image).cuda()

        pri = priori_knowledge(bathy_path)
        pri = np.array(pri)
        pri = np.transpose(pri,(1,2,0))
        pri = transforms.ToTensor()(pri)
        pri = pri.to(torch.float32).cuda()

        label = torch.from_numpy(np.asarray(read_label(label_path), dtype=np.int32)).long().cuda()
        pri_label = torch.from_numpy(np.asarray(read_seafloor(seafloor_path), dtype=np.int32)).long().cuda()

        return image, label, label_path, pri, pri_label

    def __len__(self):
        return len(self.backs)

