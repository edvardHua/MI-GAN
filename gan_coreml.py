#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 18:13
# @Author  : zengzihua
# @File    : gan_coreml.py


import os
import cv2
import numpy as np
import torch
import coremltools as ct
from copy import deepcopy


class ImageInpainter:
    def __init__(self, model_path="migan.mlmodel"):
        """
        初始化图像修复器
        Args:
            model_path: CoreML模型的路径
        """
        self.model = ct.models.MLModel(model_path)
        self.target_size = (512, 512)

    def _obtain_bbox_by_mask(self, img):
        # 检查是否读取成功
        if img is None:
            raise ValueError("无法读取图像，请检查文件路径是否正确")

        # 获取图像的宽高
        img_height, img_width = img.shape[:2]

        # 查找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果没有找到轮廓，返回 None
        if len(contours) == 0:
            print("未找到任何分割区域")
            return None

        # 找到最大的轮廓（假设分割区域为最大的轮廓）
        largest_contour = max(contours, key=cv2.contourArea)

        # 获取该轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)

        # TODO: 只有当这个轮廓面积小于原面积 1/2 时，才做 crop 吧
        if (w * h) > (img_height * img_width * 0.5):
            return 0, 0, img_width, img_height

        # 计算当前 bounding box 的中心点
        center_x = x + w // 2
        center_y = y + h // 2

        # 计算目标大小的一半
        target_width, target_height = self.target_size
        half_target_width = target_width // 2
        half_target_height = target_height // 2

        # 计算扩展后的 bounding box 的坐标（确保不超过图像边界）
        x_start = max(0, center_x - half_target_width)
        new_w = min(target_width, img_width - x_start)
        x_end = x_start + new_w
        while new_w < target_width:
            x_start = max(0, x_start - 1)
            new_w = x_end - x_start
            if new_w == target_width: break
            x_end = min(img_width, x_end + 1)
            new_w = x_end - x_start

        y_start = max(0, center_y - half_target_height)
        new_h = min(target_height, img_height - y_start)
        y_end = y_start + new_h
        while new_h < target_height:
            y_start = max(0, y_start - 1)
            new_h = y_end - y_start
            if new_h == target_height: break
            y_end = min(img_height, y_end + 1)
            new_h = y_end - y_start
        return x_start, y_start, x_end, y_end

    def _predict_by_coreml(self, image, mask):
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        # 这里 invert 一下
        mask = 255 - mask
        mask[mask < 255] = 0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ori = deepcopy(image)
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, :, :, :]
        image = image * 2.0 / 255. - 1.0
        mask = mask / 255.0
        mask = mask[np.newaxis, np.newaxis, :, :]

        image_masked = image * mask

        input_ndarr = np.concatenate([mask - 0.5, image_masked], axis=1, dtype=np.float32)
        input_ndarr.tofile("input_data_gt.npy")

        input_data = {
            "input": input_ndarr
        }

        output = self.model.predict(input_data)
        # 设置了返回值是图片，所以这里是4通道，也就是 bgra
        output = output['output']
        output = np.array(output)
        output_tmp = output[:,:,:3].astype(np.uint8)
        # 这里已经是inpaint的结果了
        cv2.imwrite("test_out.png", output_tmp[:,:,::-1])

        mask = mask.squeeze()
        mask = mask[:,:,np.newaxis]
        comp_imgs = mask * image_ori + (1-mask)*output[:,:,:3]
        return comp_imgs[:,:,::-1].astype(np.uint8)

        # output = self.model.predict(input_data)
        # output = output['var_1268']
        # comp_imgs = mask * image + (1 - mask) * output
        # comp_imgs = comp_imgs.squeeze()
        # comp_imgs = np.transpose(comp_imgs, (1, 2, 0))
        # comp_imgs = (comp_imgs * 0.5 + 0.5) * 255.0
        # return comp_imgs[:, :, ::-1].astype(np.uint8)

    def inpaint(self, image_path, mask_path, save_path=None):
        """
        执行图像修复
        Args:
            image_path: 输入图像路径
            mask_path: 掩码图像路径
            save_path: 保存结果的路径（可选）
        Returns:
            修复后的图像
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError("无法读取图像，请检查文件路径是否正确")

        h, w, _ = image.shape
        if h in range(513, 1300) and w in range(513, 1300):
            bbox = self._obtain_bbox_by_mask(mask)
            image_copy = deepcopy(image)
            cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.imwrite("image_with_box.jpg", image_copy)
            crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crop_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            out = self._predict_by_coreml(crop_image, crop_mask)
            image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = out
        else:
            # 如果图像尺寸不在指定范围内，直接处理整个图像
            out = self._predict_by_coreml(image, mask)
            image = out

        if save_path:
            cv2.imwrite(save_path, image)

        return image


if __name__ == "__main__":
    inpainter = ImageInpainter("migan.mlpackage")
    result = inpainter.inpaint(
        image_path="/Users/bigo/Workspace/InpaintApp/Inpaint/Resource/crop_masked_img.jpg",
        mask_path="/Users/bigo/Workspace/InpaintApp/Inpaint/Resource/crop_mask.jpg",
        save_path="test3.png"
    )
    pass
