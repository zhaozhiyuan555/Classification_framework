import cv2

# 读取两张图片
import numpy as np

image1 = cv2.imread(r"D:\Python\company\Target_classification\PyTorch-Classification-Trainer\data\test_images\rubbish\O_12568.jpg")
image2 = cv2.imread(r"D:\Python\company\Target_classification\PyTorch-Classification-Trainer\data\test_images\rubbish\R_10002.jpg")

# 将每张图片调整到相同的尺寸
image1 = cv2.resize(image1, (225, 225))
image2 = cv2.resize(image2, (225, 225))
# 水平拼接
horizontal_merge = cv2.hconcat([image1, image2])

# 垂直拼接
vertical_merge = cv2.vconcat([image1, image2])
cv2.imshow("img", horizontal_merge)
cv2.waitKey(0)
# 保存拼接后的图片
cv2.imwrite(r"D:\Python\company\Target_classification\PyTorch-Classification-Trainer\data\test_images\rubbish\OR_1.jpg", horizontal_merge)
cv2.imwrite(r"D:\Python\company\Target_classification\PyTorch-Classification-Trainer\data\test_images\rubbish\OR_2.jpg", vertical_merge)
