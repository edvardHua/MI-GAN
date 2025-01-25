#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 12:02
# @Author  : zengzihua
# @File    : test.py


import cv2
import numpy as np
import torch
import coremltools as ct
from lib.model_zoo.migan_inference import Generator as MIGAN
from PIL import Image

def draw_bounding_box_on_image(binary_image_path, output_image_path):
    # 读取二值化分割图
    img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # 检查是否读取成功
    if img is None:
        raise ValueError("无法读取图像，请检查文件路径是否正确")

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

    # 读取原图以绘制边界框
    original_img = cv2.imread(binary_image_path)

    if original_img is None:
        raise ValueError("无法读取原图，请检查文件路径是否正确")

    # 在原图上绘制红色的边界框 (BGR: 0, 0, 255 for red)
    cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 保存带有边界框的图像
    cv2.imwrite(output_image_path, original_img)

    print(f"Bounding Box 已绘制，并保存为: {output_image_path}")


def expand_bounding_box(binary_image_path, target_size=(512, 512)):
    # 读取二值化分割图
    img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

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

    # 计算当前 bounding box 的中心点
    center_x = x + w // 2
    center_y = y + h // 2

    # 计算目标大小的一半
    target_width, target_height = target_size
    half_target_width = target_width // 2
    half_target_height = target_height // 2

    # 计算扩展后的 bounding box 的坐标（确保不超过图像边界）
    new_x = max(0, center_x - half_target_width)
    new_y = max(0, center_y - half_target_height)
    new_w = min(target_width, img_width - new_x)
    new_h = min(target_height, img_height - new_y)

    return new_x, new_y, new_w, new_h


def test1():
    # 使用示例
    binary_image_path = 'examples/places2_512_object/masks/22.png'
    output_image_path = 'output_with_bbox.png'
    draw_bounding_box_on_image(binary_image_path, output_image_path)
    pass


def export_to_coreml(model, save_path, input_shape=(1, 4, 512, 512)):
    """
    Args:
    model (nn.Module): The MobileNetV2 model instance.
    save_path (str): Path to save the CoreML model.
    input_shape (tuple): Shape of the input tensor (default: (1, 3, 320, 320)).

    Returns:
    None
    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Convert to CoreML
    input_shape = ct.Shape(shape=input_shape)

    inputs = [ct.TensorType(name="input", shape=input_shape)]
    output_description = ct.ImageType(
        name="output",
        scale=1.0,
        bias=[0, 0, 0]
    )
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=[output_description],
        convert_to="mlprogram"
    )
    # Save the CoreML model
    coreml_model.save(save_path)

    print(f"Done export coreml model.")


def resize_image(input_image_path, output_image_path, new_width):
    # 打开图片
    with Image.open(input_image_path) as img:
        # 获取原始尺寸
        original_width, original_height = img.size

        # 计算新的高度，保持宽高比
        new_height = int((new_width / original_width) * original_height)

        # 等比例缩放图片
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 保存缩放后的图片
        resized_img.save(output_image_path)
        print(f"图片已成功保存到 {output_image_path}")


if __name__ == '__main__':
    resolution = 512
    # 导出模型时，img 会自动做缩放
    model = MIGAN(resolution=512, export=True)
    model.load_state_dict(torch.load("models/migan_512_places2.pt"))
    model.eval()
    export_to_coreml(model, "migan.mlpackage")
    # resize_image("/Users/bigo/Workspace/MagicPixel/MI-GAN/examples/places2_512_object/images/5.png",
    #              "/Users/bigo/Workspace/MagicPixel/MI-GAN/examples/places2_512_object/images/image4.png",
    #              720)
    # compare_ndarr()
    pass
