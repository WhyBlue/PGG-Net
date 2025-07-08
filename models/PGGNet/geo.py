# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
from osgeo import gdal
import matplotlib.pyplot as plt
import subprocess


def ComputeSlope(data, x_cellsize, y_cellsize):
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=int)
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=int)
    dx = cv.filter2D(data, -1, kernel_x)/(8*x_cellsize) #计算在x方向上的变化率[dz/dx]
    dy = cv.filter2D(data, -1, kernel_y)/(8*y_cellsize) #计算在y方向上的变化率[dz/dy]
    slope_degree = np.arctan(np.sqrt(dx*dx+dy*dy)) * 57.29578
    slope_degree = np.around(slope_degree, decimals=2)  #保留小数点后两位
    return slope_degree


def ComputeAspect(data, x_cellsize, y_cellsize):
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=int)
    kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=int)
    dx = cv.filter2D(data, -1, kernel_x)/8
    dy = cv.filter2D(data, -1, kernel_y)/8
    aspect = 57.29578 * np.arctan2(dy, -dx)
    slope = ComputeSlope(data, x_cellsize, y_cellsize)
    cell = aspect
    cell[np.where(aspect<0)] = 90 - aspect[np.where(aspect<0)]
    cell[np.where(aspect>90)] = 360 - aspect[np.where(aspect>90)] + 90
    cell[np.where(0<=aspect) and np.where(aspect<=90)] = 90 - aspect[np.where(0<=aspect) and np.where(aspect<=90)]
    cell[np.where(slope==0)] = -1   #零坡度的像元坡向为-1
    return cell

def hillshade(array, azimuth=90, angle_altitude=45):
    """
    :param array: input USGS ASCII DEM / CDED .dem
    :param azimuth: sun position
    :param angle_altitude: sun angle
    :return: numpy array
    """
    # 计算梯度
    x, y = np.gradient(array)
    # 计算坡度
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    # 计算坡向
    aspect = np.arctan2(-x, y)
    # 计算方位角
    azimuthrad = azimuth * np.pi / 180.


    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) \
         + np.cos(altituderad) * np.cos(slope) \
         * np.cos(azimuthrad - aspect)
    # return 255*(shaded + 1)/2
    return aspect

