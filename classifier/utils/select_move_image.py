# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-21 17:23:01
"""
import os
import cv2
import random
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


def get_file_id(file_list, postfix=".xml"):
    file_id = []
    for path in file_list:
        id = os.path.basename(path)[:-len(postfix)]
        file_id.append(id)
    return file_id


def select_image_dir_search(target_dir, search_dir, dest_dir):
    """
    从目标路径target_dir搜素search_dir，匹配ID一致的文件，并拷贝到dest_dir
    :param target_dir:
    :param search_dir:
    :param dest_dir:
    :return:
    """
    file_utils.create_dir(dest_dir)
    targer_list = file_utils.get_files_lists(target_dir, postfix=["*.png", "*.jpg"])
    targer_id = get_file_id(targer_list)
    search_list = file_utils.get_files_lists(search_dir, postfix=["*.png", "*.jpg"])

    for path in tqdm(search_list):
        search_id = os.path.basename(path)[:-len(".png")]
        if search_id in targer_id:
            print(search_id)
            src_image_file = os.path.join(search_dir, "{}.jpg".format(search_id))
            # src_image_file = os.path.join(search_dir, "{}.png".format(search_id))
            # dest_json_dir = os.path.join(dest_dir, "json")
            dest_image_dir = os.path.join(dest_dir)
            # file_processing.copy_file_to_dir(src_json_file, dest_json_dir)
            # file_processing.copy_file_to_dir(src_image_file, dest_image_dir)
            # file_utils.move_file_to_dir(src_json_file, dest_json_dir)
            file_utils.copy_file_to_dir(src_image_file, dest_image_dir)


def select_image_dir(targer_dir, match_dir, dest_dir):
    targer_list = file_utils.get_files_lists(targer_dir, postfix=["*.xml"])

    for src_xml_file in tqdm(targer_list):
        search_id = os.path.basename(src_xml_file)[:-len(".xml")]
        # src_json_file = os.path.join(match_dir, "json", "{}.json".format(search_id))
        src_image_file = os.path.join(match_dir, "{}.jpg".format(search_id))
        dest_json_dir = os.path.join(dest_dir, "json")
        dest_image_dir = os.path.join(dest_dir)
        # file_processing.copy_file_to_dir(src_json_file, dest_json_dir)
        # file_processing.copy_file_to_dir(src_image_file, dest_image_dir)
        # file_processing.move_file_to_dir(src_json_file, dest_json_dir)
        file_utils.move_file_to_dir(src_image_file, dest_image_dir)


def select_orientation_images(targer_dir, match_dir, dest_dir):
    targer_list = file_utils.get_files_lists(targer_dir, postfix=["*.xml"])
    for src_xml_file in tqdm(targer_list):
        search_id = os.path.basename(src_xml_file)[:-len(".xml")]
        # src_json_file = os.path.join(match_dir, "json", "{}.json".format(search_id))
        src_image_file = os.path.join(match_dir, "{}.jpg".format(search_id))
        image = image_utils.read_image(src_image_file)
        h, w, d = image.shape
        if w >= h:  # 横屏
            # dest_json_dir = os.path.join(dest_dir, "json")
            dest_image_dir = os.path.join(dest_dir, "landscape/images")
            dest_xml_dir = os.path.join(dest_dir, "landscape/annotations/xml")
        else:
            # 竖屏
            # dest_json_dir = os.path.join(dest_dir, "json")
            dest_image_dir = os.path.join(dest_dir, "portrait/images")
            dest_xml_dir = os.path.join(dest_dir, "portrait/annotations/xml")
        file_utils.copy_file_to_dir(src_image_file, dest_image_dir)
        file_utils.copy_file_to_dir(src_xml_file, dest_xml_dir)


def select_sub_image_dir(image_root, dst_root, max_nums=10, shuffle=True):
    """实现移动子文件里面的图片"""
    sub_list = file_utils.get_sub_paths(image_root)
    # sub_file = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval/class_name699.txt"
    # sub_list = file_utils.read_data(sub_file, split=None)
    for sub in tqdm(sub_list):
        image_dir = os.path.join(image_root, sub)
        image_list = file_utils.get_files_lists(image_dir)
        if shuffle:
            random.seed(100)
            random.shuffle(image_list)
        if max_nums > 0:
            num = min(max_nums, len(image_list))
            image_list = image_list[:num]
        for image_path in image_list:
            dst_dir = os.path.join(dst_root, sub)
            # file_utils.copy_file_to_dir(image_path, dst_dir)
            file_utils.move_file_to_dir(image_path, dst_dir)


def convert_image_format(image_dir, postfix="png"):
    """转换图片格式"""
    image_list = file_utils.get_files_lists(image_dir)
    for image_path in image_list:
        dirname = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        image_id = basename.split(".")
        assert len(image_id) == 2
        image_id = image_id[0]
        outpath = os.path.join(dirname, "{}.{}".format(image_id, postfix))
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            cv2.imwrite(outpath, alpha)
        else:
            print(image_path)


def rename_sub_file_dir(src_dir, flag="matting_"):
    data_list = file_utils.get_sub_directory_list(src_dir)
    for name in data_list:
        sub_dir = file_utils.create_dir(src_dir, name)
        sub_list = file_utils.get_sub_directory_list(sub_dir)
        for sub in sub_list:
            sub_path = file_utils.create_dir(sub_dir, sub)
            if flag in sub:
                new_sub = sub[len(flag):]
                new_path = file_utils.create_dir(sub_dir, new_sub)
                os.rename(sub_path, new_path)
                print(new_path)


def demo_select_image():
    targer_dir = "/home/dm/data3/dataset/finger_keypoint/finger/val/annotations/val"
    search_dir = "/home/dm/data3/dataset/finger_keypoint/finger/images"
    dest_dir = "/home/dm/data3/dataset/finger_keypoint/finger/val/images"
    # select_image_dir_search(targer_dir, search_dir, dest_dir)
    select_image_dir(targer_dir, search_dir, dest_dir)


if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/animal/animals10/train"
    dst_root = "/home/dm/nasdata/dataset/csdn/animal/animals10/test"
    select_sub_image_dir(image_dir, dst_root, max_nums=100, shuffle=True)
