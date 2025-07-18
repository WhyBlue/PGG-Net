import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.PGGNet import net
from utils.unet_dataset import read_tiff_3,priori_knowledge,read_seafloor
from osgeo import gdal
from metrics import eval_metrics
from train import toString
import torch


def read_label(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset
    return im_data


def eval(config, round):
    device = torch.device('cuda:0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    selected = config['train_model']['model'][config['train_model']['select']]
    if selected == 'PGGNet':
        model = net.PGGNet(num_classes=config['num_classes'])

    check_point = os.path.join(config['save_model']['save_path'] + selected + '/' + round, selected + '.pth')
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.614, 0.593], std=[0.059, 0.010])
        ]
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    # 混淆矩阵
    conf_matrix_test = np.zeros((config['num_classes'], config['num_classes']))

    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = 0.0
    unoin_sum = 0.0
    pixelAcc = 0.0
    mIoU = 0.0

    class_precision = np.zeros(config['num_classes'])
    class_recall = np.zeros(config['num_classes'])
    class_f1 = np.zeros(config['num_classes'])
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            back_path, bathy_path, seafloor_path, label_name = line.strip().split()
            root_dir = 'Dataset/'
            back_path = os.path.join(root_dir, back_path).replace('\\', '/')
            bathy_path = os.path.join(root_dir, bathy_path).replace('\\', '/')
            seafloor_path = os.path.join(root_dir, seafloor_path).replace('\\', '/')
            label_name = os.path.join(root_dir, label_name).replace('\\', '/')
            label = torch.from_numpy(np.asarray(read_label(label_name), dtype=np.int32)).long().cuda()
            pri_label = torch.from_numpy(np.asarray(read_seafloor(seafloor_path), dtype=np.int32)).long().cuda()

            image = read_tiff_3(back_path=back_path, bathy_path=bathy_path, is_train=True)
            image = np.array(image)
            image = np.transpose(image, (1, 2, 0))
            image = transforms.ToTensor()(image)
            image = image.to(torch.float32).cuda()
            image = transform(image).cuda()
            # 加一维,batch_size=1
            image = image.unsqueeze(0)

            pri = priori_knowledge(bathy_path)
            pri = np.array(pri)
            pri = np.transpose(pri, (1, 2, 0))
            pri = transforms.ToTensor()(pri)
            pri = pri.to(torch.float32).cuda()
            pri = pri.unsqueeze(0)

            output,pri_output = model(image, pri)
            # _, pred = output.max(1)
            # pred = pred.view(256, 256)
            # mask_im = pred.cpu().numpy().astype(np.uint8)
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label, config['num_classes'],
                                                                            conf_matrix_test)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

            for i in range(config['num_classes']):
                # 每一类的precision
                class_precision[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[:, i].sum()
                # 每一类的recall
                class_recall[i] = 1.0 * conf_matrix_test[i, i] / conf_matrix_test[i].sum()
                # 每一类的f1
                class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
    print('OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
        pixelAcc, toString(mIoU), mIoU.mean(), toString(class_precision), toString(class_recall), toString(class_f1)))

    if not os.path.isdir('saved/' + selected + '/' + round + "/confuse_matrix"):
        os.makedirs('saved/' + selected + '/' + round + "/confuse_matrix")
    m = open(os.path.join('saved/' + selected + '/' + round + "/confuse_matrix", selected + '_result_test.txt'), 'w')
    m.write('OA:' + str(pixelAcc) + '\n')
    m.write('IOU:' + toString(mIoU) + '\n')
    m.write('mIoU:' + str(mIoU.mean()) + '\n')
    m.write('class_precision:' + toString(class_precision) + '\n')
    m.write('class_recall:' + toString(class_recall) + '\n')
    m.write('class_f1:' + toString(class_f1) + '\n')

    np.savetxt(os.path.join('saved/' + selected + '/' + round + "/confuse_matrix", selected + '_matrix_test.txt'),
               conf_matrix_test, fmt="%d")


if __name__ == "__main__":
    with open(r'eval_config.json', encoding='utf-8') as f:
        config = json.load(f)
    # eval(config)
    for round in config['train_model']['round']:
        eval(config, round)