# -*-coding: utf-8 -*-
import os
import sys

sys.path.append("libs")
import argparse
import PIL.Image as Image
from basetrainer.utils import log, setup_config
from pybaseutils import file_utils, image_utils, coords_utils
from classifier import inference


class Predictor(inference.Inference):
    def __init__(self, cfg):
        super(Predictor, self).__init__(cfg)

    def predict(self, faces):
        pred_index, pred_score = self.inference(faces, cfg.threshold)
        pred_index = self.label2class_name(pred_index)
        return pred_index, pred_score

    def image_dir_predict(self, image_dir, save_dir, isshow=True, shuffle=True):
        image_list = file_utils.get_files_lists(image_dir, shuffle=shuffle,
                                                postfix=["*.jpg", "*.jpeg", "*.webp", "*.png"])
        for path in image_list:
            image = image_utils.read_image(path, use_rgb=True)
            pred_index, pred_score = self.predict(image)
            info = "path:{} pred_index:{},pred_score:{}".format(path, pred_index, pred_score)
            print(info)
            if isshow:
                # image_utils.cv_show_image("predict", image, use_rgb=True)
                image = image_utils.draw_text_pil(image, (10, 20), str(pred_index), size=20, color_color=(255, 255, 255))
                # image_utils.cv_show_image("predict1", image, use_rgb=True)

            if save_dir:
                print("保存文件的路径：",os.path.join(save_dir,str(path).split("\\")[-1]))
                image_utils.save_image(os.path.join(save_dir,str(path).split("\\")[-1]),image)

def get_parser():
    # 配置文件
    config_file = "work_space/mobilenet_v2_1.0_CrossEntropyLoss_20230913_144644_1362/config.yaml"
    # 模型文件
    model_file = "work_space/mobilenet_v2_1.0_CrossEntropyLoss_20230913_144644_1362/model/best_model_000_97.6562.pth"
    # 待测试图片目录
    image_dir = "data/test_images/test1"
    save_dir = "data/test_images/result3"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parser = argparse.ArgumentParser(description="Inference Argument")
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument("-m", "--model_file", help="model_file", default=model_file, type=str)
    parser.add_argument("-i", "--isshow", help="show pic", default=True, type=bool)
    parser.add_argument("-t", "--threshold", help="roi阈值", default=0.4, type=float)
    parser.add_argument("--device", help="cuda device id", default="cpu", type=str) # cuda:0
    parser.add_argument("--image_dir", help="image file or directory", default=image_dir, type=str)
    parser.add_argument("--save_dir", help="image result save file", default=save_dir, type=str)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    cfg = setup_config.parser_config(parser.parse_args(), cfg_updata=False)
    t = Predictor(cfg)
    t.image_dir_predict(cfg.image_dir, cfg.save_dir, isshow=cfg.isshow, shuffle=False)
