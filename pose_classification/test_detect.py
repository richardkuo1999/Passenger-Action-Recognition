# This code draw ground truth bounding box 
# base on detection result images

import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets_test_detect import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=8,
         img_size=608,
         iou_thres=0.5,
         conf_thres=0.5,
         nms_thres=0.5,
         save_json=False,
         model=None):

    # Configure run
    data = parse_data_cfg(data)
    print(data)
    nc = int(data['classes'])  # number of classes

    test_path = data['valid']  # path to test images
    print(test_path)
    names = load_classes(data['names'])  # class names
    s = ('%30s' + '%10s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1', 'Acc')
    # Get classes and colors
    #classes = load_classes(data['names'])
    classes = ['sit','stand']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    dataset_path = 'E:/labeled_2/test.txt'
    images = open(dataset_path).read().strip().split()
    
    for index, image in enumerate(tqdm(images, desc='Computing mAP')):
        img_name = image.split('/')[-1]
        #img_name = image.split('_')[-1]
        img = cv2.imread('output/'+img_name)
        print(img_name)
        img_h, img_w, _ = img.shape
        ground_truth = open(image.replace('jpg', 'txt')).read().strip().split('\n')
        print(ground_truth)
        for obj in ground_truth:
            image_id = 0.
            obj = obj.split()
            cls = float(obj[0])
            x = int(float(obj[1]) * img_w)
            y = int(float(obj[2]) * img_h)
            w = int(float(obj[3]) * img_w/2)
            h = int(float(obj[4]) * img_h/2)
            label = '%s' % (classes[int(cls)])
            xyxy = [x-w,y-h,x+w,y+h]
            plot_one_box(xyxy, img, label=label, color=(0,0,255), person_count=0,pose_class='')
        cv2.imwrite('output/'+img_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        mAP = test(opt.cfg,
                   opt.data,
                   opt.weights,
                   opt.batch_size,
                   opt.img_size,
                   opt.iou_thres,
                   opt.conf_thres,
                   opt.nms_thres,
                   opt.save_json)
