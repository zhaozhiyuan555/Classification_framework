import numpy as np


def filter_array(arr, threshold):
    # 获取满足条件的索引和值
    indices = np.where(arr >= threshold)[0]
    values = arr[indices]

    return values, indices


# 示例数组
arr = np.array([1, 2, 3, 4, 5])
threshold = 3

filtered_values, filtered_indices = filter_array(arr, threshold)
print("Filtered Values:", filtered_values)
print("Filtered Indices:", filtered_indices)
