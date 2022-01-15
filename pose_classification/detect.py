import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

#for resnet
import cv2  
from PIL import Image,ImageOps 
import PIL
import numpy
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from thop import profile

#for resnet
pose_classes = ['stand','sit']
#for resnet
def letterbox(img, mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = [img.size[1],img.size[0]]  # current shape [height, width]
    new_shape = 224
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = img.resize(new_unpad,PIL.Image.ANTIALIAS)
    img = ImageOps.expand(img, border=(left,top,right,bottom), fill=(128,128,128))
    return img

#for resnet
#Override transform's resize function(using letter box)
class letter_img(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        #letterbox(img).save('out.bmp')
        return letterbox(img)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
#for resnet
transform_test = transforms.Compose([
    letter_img(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


#start of original file
def detect(cfg,
           data,
           weights,
           images='data/samples',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           save_txt=True,
           save_images=True,
           save_videos=False,
           webcam=False): #parameter to load resnet weight
    # Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = DarknetRes(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.model1.load_state_dict(torch.load(weights, map_location=device)['model1'])
        model.model2.load_state_dict(torch.load(weights, map_location=device)['model2'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    '''
    #load resnet
    net2 = models.resnet50()
    device2 = 'cuda' if torch.cuda.is_available() else 'cpu'
    net2 = net2.to(device2)
    if device2 == 'cuda':
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/ckpt.pth')
    net2.load_state_dict(checkpoint['net'])
    '''

    # Fuse Conv2d + BatchNorm2d layers
    #model.fuse()

    # Eval mode
    model.to(device).eval()
    #model.to(device)

    #model_info(model, report='full')
    #inn = torch.randn(1, 3, 512, 512).to(device)
    #flops, params = profile(model, inputs=(inn, ))
    #print('flops: ', flops, ' params: ', params)

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    
        
    #save to a list
    person_num_list = [0]*7
    person_list_index = 0
    #print(person_num_list)

    #save video parameter
    video_fir = 1
    video_sec = 2

    t_all = time.time()
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        #print(np.shape(im0))
        #print(np.shape(img))
        #print(np.shape(img))
        #cv2.imwrite('letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        save_path = str(Path(output) / Path(path).name)
        #print(save_path)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        #pred, _ = model(img)
        #det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        det, cla_res = model(img)
        #print(cla_res)
        #print(type(det))
        det = det[0]
        person_num = 0
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            #print(img.shape[2:])
            #print(im0.shape)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            #det[:, :4] = scale_coords(img_size, det[:, :4], im0.shape).round()
            #print(det[:, :4])
            # Print results to screen
            #print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            #draw person number in the image
            for label_count in range(0,len(det)):
                if det[label_count][6] == 0 or det[label_count][6] == 1:
                    person_num = person_num+1
            tl_2 = round(0.002 * (im0.shape[0] + im0.shape[1]))
            tf = max(tl_2 - 1, 1)

            
            #person_num = str(person_num)
            person_num_list[person_list_index] = person_num
            if person_list_index==6:
                person_list_index = 0
            else:
                person_list_index = person_list_index+1
            #print(person_num_list, end=', ')
            person_num_result = max(person_num_list)
            
            #person_num_result = 0
            im0 = cv2.putText(im0, 'num='+str(person_num_result) , (50,50), 0, tl_2/3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
            # Draw bounding boxes and labels of detections
            person_count = person_num_result
            det_num = 0
            #print(det[0][0],det[0][1])
            for *xyxy, conf, cls_conf, _ in det:
                #det_num = det_num+1
                #print('\npath:')
                pp = path.split('\\')[-1]
                pp = pp.split('.')[0]
                #print(pp)
                '''
                # Save txt( only person class)
                if save_txt:  # Write to file
                    with open('person.txt', 'a') as file:
                        xyxy = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        file.write(('%s '*1 + '%g ' * 5 + '\n') % (pp, conf, *xyxy))
                '''
                
                if save_txt:  # Write to file
                    with open(output+'/sit.txt', 'a') as file:
                        if classes[int(cla_res[det_num])]=='sit':
                            xyxy = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            #xyxy = int(xyxy)
                            file.write(('%s '*1 + '%g ' * 5 + '\n') % (pp, conf, *xyxy))
                    with open(output+'/stand.txt', 'a') as file2:
                        if classes[int(cla_res[det_num])]=='stand':
                            xyxy = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            #xyxy = int(xyxy)
                            file2.write(('%s '*1 + '%g ' * 5 + '\n') % (pp, conf, *xyxy))
                

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cla_res[det_num])], conf)
                '''
                if classes[int(cls)] == 'sit' or classes[int(cls)] == 'stand':

                    
                    #for two stage
                    temp = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    #cv2.imwrite('gg.jpg',temp)
                    cropped = Image.fromarray(cv2.cvtColor(temp,cv2.COLOR_BGR2RGB))
                    cropped_tensor = transform_test(cropped)
                    cropped_tensor.unsqueeze_(0)
                    cropped_variable = Variable(cropped_tensor)
                    classify_result = net2(cropped_variable)
                    re_class = classify_result.argmax()
                    re_class = pose_classes[re_class.item()]
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], person_count=person_count,pose_class=re_class)
                    person_count = person_count+1
                    
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], person_count=person_count,pose_class='')
                    #person_count = person_count+1
                '''
                plot_one_box(xyxy, im0, label=label, color=colors[int(cla_res[det_num])], person_count=person_count,pose_class='')
                det_num = det_num+1

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                #print(im0)
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)
        #save images(dataset) to video
        if save_videos:
            if video_fir != video_sec:  # new video
                video_fir = video_sec
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                fps = 6
                width = 2704
                height = 1520
                #width = 2026
                #height = 1520
                #width = 2048
                #height = 1536
                vid_writer = cv2.VideoWriter('test_video.avi', cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)

    print(time.time()-t_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/obj.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/actstand2', help='path to images')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
