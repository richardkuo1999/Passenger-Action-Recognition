'''
Two camera detection & Add overlapped area association version
Main camera is 'cam2'
Sub camera is 'cam1'
v5 初步整合姿態跟動作的分類結果(開會紀錄0623)
邊界框預測xywh in 0~1(非0~99)
'''
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

from hungarian import *
from ioumatch import *
from imgsaver import *
#from utils import resnet_lgdv2 as lgd
from utils import resnet_2p1d_lgdv2_cbam as lgd
from PIL import Image
from spatial_transforms_winbus import *

from overlapmodel import Net6 as overlapmodel
from bbmatch import *


# To fixed scale of person
def letterbox(img, resize_size=112, mode='square'):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    shape = [img.size[1],img.size[0]]  # current shape [height, width]
    new_shape = resize_size
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
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
    #img = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)
    return img
#Override transform's resize function(using letterbox)
class letter_img(transforms.Resize):
    #def __init__(self, size, interpolation=Image.BILINEAR):
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

# Params for action classifier
mean = [114.7748, 107.7354, 99.475]
std = [1, 1, 1]
norm_method = Normalize(mean, std)

'''
# Params for overlapped area association for winbus-1002 未估計前
thres_upbb_up = 100#47 #100
thres_upbb_down = 200#94 #200
thres_down = 420#355 #750
chair_down = 450 
'''
drawRegion = False
def getbbinarea(det_1, det_2, side, matchindex, img, imgw, showid): # Select box in overlap area
    '''
    det_1: [detection number,4], xywh in 0~1
    det_2: [detection number, 6], x1y1x2y2 in pixel position
    side: which side is chair, {'left','right'}
    matchindex: to continue count people in two camera
    img: original size image
    imgw: img size of width
    showid: to draw box not in overlap area
    The overlap area projected to image:
    horizontal line in 2704*1520 resolution: 171, 385, 815
    vertical line in 2704*1520 resolution: 1063 in two side
    '''
    lowreso = True # If the input resolution is 2704*1520 set false, if input resolution is 1280*720 set true.
    thres_upbb_up = int(171 * 0.473688) if lowreso else 171 #69 222
    thres_upbb_down = int(385 * 0.473688) if lowreso else 385 #385
    thres_down = int(815 * 0.473688) if lowreso else 815
    chair_down = int(1063 * 0.47337) if lowreso else 1063
    bbinarea1 = None # box info: xywh in 0~1
    bbinarea1_2 = None # box info: x1y1x2y2 in pixel position
    notinarea1 = 0

    for i in range(len(det_2)):
        upbb_up = thres_upbb_up > det_2[i][1]
        upbb_down = thres_upbb_down < det_2[i][3]
        down = det_2[i][3] < thres_down
        if side == 'left':
            chairbottom =  upbb_up and upbb_down and (det_1[i][0] < chair_down/imgw)
        else:
            chairbottom = upbb_up and upbb_down and (det_1[i][0] > (imgw-chair_down)/imgw)
        #print(upbb_up,upbb_down,down,chairbottom)
        if upbb_up and upbb_down and down or chairbottom:
            if bbinarea1 is None:
                bbinarea1 = det_1[i].unsqueeze(0)
                if showid:
                    bbinarea1_2 = det_2[i].unsqueeze(0)
            else:
                bbinarea1 = torch.cat((bbinarea1, det_1[i].unsqueeze(0)),0)
                if showid:
                    bbinarea1_2 = torch.cat((bbinarea1_2, det_2[i].unsqueeze(0)),0)
        elif upbb_up and (not upbb_down):
            pass
        else:
            notinarea1 += 1
            if showid:
                label = 'bb id:%d' % (matchindex)
                plot_one_box(det_2[i][:4], img, label=label, color=(255,0,0), person_count=0,pose_class='')
                matchindex += 1
    #print('-----------')
    return bbinarea1, bbinarea1_2, matchindex, notinarea1

# Params for overlapped area association for winbus-1002
thres_upbb_up = 81 #100
thres_upbb_down = 183 #200
thres_down = 386
chair_down = 503
'''
# Params for overlapped area association for winbus-1203
thres_upbb_up = 47 #100
thres_upbb_down = 94 #200
thres_down = 355
chair_down = 450
'''

#start of original file
def detect(cfg,
           data,
           weights,
           images1='data/samples',  # input folder
           images2='data/samples',  # input folder
           output1='output1',  # output folder
           output2='output2',  # output folder
           fourcc='mp4v',  # video codec
           img_size=512,
           conf_thres=0.5,
           nms_thres=0.5,
           save_txt=False,
           save_images=True,
           save_videos=True,
           webcam=False): #parameter to load resnet weight
    cor_count = 0
    # Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output1):
        shutil.rmtree(output1)  # delete output folder
    os.makedirs(output1)  # make new output folder
    if os.path.exists(output2):
        shutil.rmtree(output2)  # delete output folder
    os.makedirs(output2)  # make new output folder
    if os.path.exists('detres'):
        shutil.rmtree('detres')  # delete output folder
    os.makedirs('detres')

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

    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer1, vid_writer2 = None, None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images1, images2, img_size=img_size)

    # Get classes and colors
    #classes = load_classes(parse_data_cfg(data)['names'])
    classes = ['sitting', 'standing', 'sit', 'stand'] # action classes
    classes_pose = ['sit', 'stand']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    #save video parameter
    video_fir = 1
    video_sec = 2
    
    first_frame = True
    first_frame2 = True
    prematch_list = []
    prematch_list2 = []
    match_list = []
    match_list2 = []

    # Temporal action classification model
    actmodel = lgd.resnet50(
                num_classes=4,
                shortcut_type='B',
                sample_size=112,
                sample_duration=15).to('cuda')
    actmodel = nn.DataParallel(actmodel, device_ids=None)
    checkpoint = torch.load('weights/best-06-lgd-2p1d-cbam-957.pth')
    actmodel.load_state_dict(checkpoint['state_dict'])
    actmodel.eval()

    imgsave = Saved_imgs()
    imgsave2 = Saved_imgs()
    spatial_transform = Compose([
            letter_img(112),
            ToTensor(1),
            norm_method
        ])

    # Overlapped area assiciation model
    olareamodel = overlapmodel().to(device)
    #checkpoint = torch.load('utils/best-trainwithwinbus01_v2.pth') # For pred 0~9
    checkpoint = torch.load('weights/best-match-985.pth')
    olareamodel.load_state_dict(checkpoint['state_dict'])
    olareamodel.eval()
    softmax = nn.Softmax(dim=1)
    showid = False

    indexx = 0
    indexx2 = 0
    pred_act = None
    pred_act_prob = None
    pred_act2 = None
    pred_act2_prob = None
    batchid = []
    batchid2 = []
    t_all = time.time()

    test_inf_time = 0
    test_det_time = 0
    test_match_time = 0
    test_actck_time = 0
    test_act_time = 0
    test_overlap_time = 0
    errco = []
    inddd = 1
    inddd2 = 1
    for i, (path1, path2, img, im0, img2, im02, vid_cap) in enumerate(dataloader):
        #print('----------------------------')
        t = time.time()
        save_path1 = str(Path(output1) / Path(path1).name)
        save_path2 = str(Path(output2) / Path(path2).name)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        img2 = torch.from_numpy(img2).unsqueeze(0).to(device)
        imgs = torch.cat((img, img2),0)

        t_det = time.time()
        det, cla_res, inf_out = model(imgs, conf_thres1=conf_thres, nms_thres1=nms_thres)
        inf_out = softmax(inf_out)

        # image 1 detection results
        det1 = det[0].clone()
        cla_res1 = cla_res[:len(det1)]
        inf_out1 = inf_out[:len(det1)]

        # image 2 detection results
        det2 = det[1].clone()
        cla_res2 = cla_res[len(det1):]
        inf_out2 = inf_out[len(det1):]

        test_det_time += (time.time()-t_det)

        im0_save = im0.copy()
        im02_save = im02.copy()
        # Deal with image 1 detection
        if det1 is not None and len(det1) > 0:
            # Rescale boxes from 512 to true image size
            det1[:, :4] = scale_coords(img.shape[2:], det1[:, :4], im0.shape).round()          

            t_match = time.time()
            # Matching with iou
            temp_pre = [0] *len(det1)
            match_list = [0] *len(det1)
            if first_frame:
                first_frame = False
                for x in range(0,len(det1)):
                    temp_pre[x] = numpy.concatenate(([x], det1[x, :4].cpu()))
                    match_list[x] = numpy.concatenate(([x], det1[x, :4].cpu()))
                temp_pre = np.array(temp_pre)
                match_list = np.array(match_list)
                match_res = matching(temp_pre, match_list)
                prematch_list = temp_pre
            else:
                for x in range(0,len(det1)):
                    match_list[x] = numpy.concatenate(([x], det1[x, :4].cpu()))
                match_list = np.array(match_list)
                match_res = matching(prematch_list, match_list)
                prematch_list = match_list

            test_match_time += (time.time()-t_match)

            # Draw bounding boxes and labels of detections
            det_num = 0
            if not showid:
                for detnum in range(len(det1)):
                    if match_res[detnum][0] in batchid:
                        pa = pred_act[batchid.index(match_res[detnum][0])]
                        act_prob = pred_act_prob[batchid.index(match_res[detnum][0]), pa]
                        mix = pred_act_prob.clone()
                        mix[batchid.index(match_res[detnum][0]), 2:] = (mix[batchid.index(match_res[detnum][0]), 2:]+inf_out1[detnum])/2
                        _, mix_res = mix.max(1)
                        #label = '%s , %d, %s' % (classes_pose[int(cla_res1[detnum])], match_res[detnum][0], classes[pa])
                        label_for_txt = '%s , %.3f, %s, %.3f, %s' % (classes_pose[int(cla_res1[detnum])], inf_out1[detnum, cla_res1[detnum]], classes[pa], act_prob, classes[mix_res[batchid.index(match_res[detnum][0])]])
                        label = '%s,%3f' % (classes[pa], act_prob) # show action
                        label = '%s , %.3f' % (classes_pose[int(cla_res1[detnum])], inf_out1[detnum, cla_res1[detnum]])
                        #plot_one_box( match_res[detnum][1:], im0, label=label, color=colors[int(cla_res1[detnum])], person_count=0,pose_class='')
                        plot_one_box( match_res[detnum][1:], im0, label=label, color=colors[int(pa)], person_count=0,pose_class='') # For show action result
                        with open('detres/'+classes[pa]+'.txt', 'a') as file:
                            fn = path1.split('\\')[-1].split('.')[0].replace(' ', '_')
                            xyxy = int(match_res[detnum][1]), int(match_res[detnum][2]), int(match_res[detnum][3]), int(match_res[detnum][4])
                            #xyxy = int(batchxyxy[batchid.index(match_res[detnum][0])][0]), int(batchxyxy[batchid.index(match_res[detnum][0])][1]),\
                            #int(batchxyxy[batchid.index(match_res[detnum][0])][2]), int(batchxyxy[batchid.index(match_res[detnum][0])][3])
                            file.write(('%s '*1 + '%g ' * 1 + '%d ' * 4 + '\n') % (fn, act_prob, *xyxy))
                    else:
                        label = '%s , %d' % (classes_pose[int(cla_res1[detnum])], match_res[detnum][0])
                        #plot_one_box( match_res[detnum][1:], im0, label=label, color=colors[int(cla_res1[detnum])], person_count=0,pose_class='')

            # After tracking(with id), crop img and save in dict
            batchact = None
            batchid = []
            t_actck = time.time()
            actionofid = None
            for *xyxy, conf, cls_conf, _ in det1:   
                id = match_res[det_num][0]
                c1, c2 = (int(match_res[det_num][1]), int(match_res[det_num][2])), (int(match_res[det_num][3]), int(match_res[det_num][4]))
                cropimg = im0_save[c1[1]:c2[1], c1[0]:c2[0]]
                cropimg = spatial_transform(cropimg)
                imgsave.update(cropimg, id, indexx)
                indexx+=1
                actionofid = imgsave.checklen(id)

                if actionofid is not None:
                    if batchact is not None:
                        batchact = torch.cat((batchact, actionofid),0)
                        batchid.append(id)
                    else:
                        batchact = actionofid
                        batchid.append(id)
                det_num = det_num+1

            # del missing id data
            cur_id = [m[0] for m in match_res]
            imgsave.checkdel(cur_id)

            test_actck_time += (time.time()-t_actck)

        # Deal with image 2 detection
        if det2 is not None and len(det2) > 0:
            # Rescale boxes from 512 to true image size
            det2[:, :4] = scale_coords(img2.shape[2:], det2[:, :4], im02.shape).round()          

            t_match = time.time()
            #Matching with iou
            temp_pre2 = [0] *len(det2)
            match_list2 = [0] *len(det2)
            if first_frame2:
                first_frame2 = False
                for x in range(0,len(det2)):
                    temp_pre2[x] = numpy.concatenate(([x], det2[x, :4].cpu()))
                    match_list2[x] = numpy.concatenate(([x], det2[x, :4].cpu()))
                temp_pre2 = np.array(temp_pre2)
                match_list2 = np.array(match_list2)
                match_res2 = matching(temp_pre2, match_list2)
                prematch_list2 = temp_pre2
            else:
                for x in range(0,len(det2)):
                    match_list2[x] = numpy.concatenate(([x], det2[x, :4].cpu()))
                match_list2 = np.array(match_list2)
                match_res2 = matching(prematch_list2, match_list2)
                prematch_list2 = match_list2

            test_match_time += (time.time()-t_match)
            # Draw bounding boxes and labels of detections
            det_num = 0
            if not showid:
                for detnum2 in range(len(det2)):
                    if match_res2[detnum2][0] in batchid2:
                        pa = pred_act2[batchid2.index(match_res2[detnum2][0])]
                        act_prob2 = pred_act2_prob[batchid2.index(match_res2[detnum2][0]), pa]
                        mix2 = pred_act2_prob.clone()

                        mix2[batchid2.index(match_res2[detnum2][0]), 2:] = (mix2[batchid2.index(match_res2[detnum2][0]), 2:]+inf_out2[detnum2])/2
                        _, mix_res2 = mix2.max(1)
                        #label = '%s , %d, %s' % (classes_pose[int(cla_res2[detnum2])], match_res2[detnum2][0], classes[pa])
                        #label = '%s , %.3f, %s, %.3f, %s' % (classes_pose[int(cla_res2[detnum2])], inf_out2[detnum2, cla_res2[detnum2]], classes[pa], act_prob2, classes[mix_res2[batchid2.index(match_res2[detnum2][0])]])
                        #label = '%s , %.3f' % (classes_pose[int(cla_res2[detnum2])], inf_out2[detnum2, cla_res2[detnum2]])
                        label = '%s,%.3f' % (classes[pa], act_prob) # show action
                        #plot_one_box( match_res2[detnum2][1:], im02, label=label, color=colors[int(cla_res2[detnum2])], person_count=0,pose_class='')
                        plot_one_box( match_res2[detnum2][1:], im02, label=label, color=colors[int(pa)], person_count=0,pose_class='') # For show action result
                    else:
                        label = '%s , %d' % (classes_pose[int(cla_res2[detnum2])], match_res2[detnum2][0])
                        #plot_one_box( match_res2[detnum2][1:], im02, label=label, color=colors[int(cla_res2[detnum2])], person_count=0,pose_class='')

            # After tracking(with id), crop img and save in dict
            batchact2 = None
            batchid2 = []
            t_actck = time.time()
            actionofid2 = None
            for *xyxy, conf, cls_conf, _ in det2:
                id = match_res2[det_num][0]
                c1, c2 = (int(match_res2[det_num][1]), int(match_res2[det_num][2])), (int(match_res2[det_num][3]), int(match_res2[det_num][4]))
                cropimg = im02_save[c1[1]:c2[1], c1[0]:c2[0]]
                cropimg = spatial_transform(cropimg)
                imgsave2.update(cropimg, id, indexx)
                indexx+=1
                actionofid2 = imgsave2.checklen(id)

                if actionofid2 is not None:
                    if batchact2 is not None:
                        batchact2 = torch.cat((batchact2, actionofid2),0)
                        batchid2.append(id)
                    else:
                        batchact2 = actionofid2
                        batchid2.append(id)
                det_num = det_num+1

            # del missing id
            cur_id2 = [m[0] for m in match_res2]
            imgsave2.checkdel(cur_id2)

            test_actck_time += (time.time()-t_actck)

        t_act = time.time()
        # Do action classification in one batch
        if batchact is not None and batchact2 is not None: # Do pred with cam1 and cam2 people
            all_batchact = torch.cat((batchact, batchact2),0).cuda()
            #all_batchact = Variable(all_batchact, volatile=True).cuda()
            outputs = actmodel(all_batchact)
            _, all_pred_act = outputs.max(1)
            all_pred_prob = softmax(outputs)
            pred_act = all_pred_act[:len(batchact)]
            pred_act_prob = all_pred_prob[:len(batchact)]
            pred_act2 = all_pred_act[len(batchact):]
            pred_act2_prob = all_pred_prob[len(batchact):]
        elif batchact is not None and batchact2 is None: # Do pred with cam1 people
            outputs = actmodel(batchact.cuda())
            _, pred_act2 = outputs.max(1)
            pred_act2_prob = softmax(outputs)
        elif batchact is None and batchact2 is not None: # Do pred with cam2 people
            outputs = actmodel(batchact2.cuda())
            _, pred_act2 = outputs.max(1)
            pred_act2_prob = softmax(outputs)
        else: all_batchact = None # No people in both cam1 and cam2(need to track certain frames)

        test_act_time += (time.time()-t_act)

        #t_overlap = time.time()

        # Count people with bb predict model
        match_res_cor = None
        match_res_err = None
        remainin = None
        remaintar = None
        imgh, imgw = im0.shape[:2]
        det1t = det1.clone()
        x11, y11 = det1t[:, 0], det1t[:, 1]
        x12, y12 = det1t[:, 2], det1t[:, 3]
        det1_2 = det1.clone() # x1y1x2y2 in pixel position
        det1[:, 0], det1[:, 1] = (x11+x12)/(2*imgw), (y11+y12)/(2*imgh)
        det1[:, 2], det1[:, 3] = abs(x11-x12)/imgw, abs(y11-y12)/imgh
        det1 = det1[:, :4] # xywh in 0~1

        det2t = det2.clone()
        x21, y21 = det2t[:, 0], det2t[:, 1]
        x22, y22 = det2t[:, 2], det2t[:, 3]
        det2_2 = det2.clone()
        det2[:, 0], det2[:, 1] = (x21+x22)/(2*imgw), (y21+y22)/(2*imgh)
        det2[:, 2], det2[:, 3] = abs(x21-x22)/(imgw), abs(y21-y22)/(imgh)
        det2 = det2[:, :4]
        matchindex = 0

        #test_overlap_time += (time.time()-t_overlap)
        t_overlap = time.time()

        # Chairs in left side
        bbinarea1, bbinarea1_2, matchindex, notinarea1 = getbbinarea(det1, det1_2, 'left', matchindex, im0, imgw, showid)

        test_overlap_time += (time.time()-t_overlap)
        #t_overlap = time.time()

        # Chairs in right side
        bbinarea2, bbinarea2_2, matchindex, notinarea2 = getbbinarea(det2, det2_2, 'right', matchindex, im02, imgw, showid)

        #t_overlap = time.time()
        #test_overlap_time += (time.time()-t_overlap)

        if bbinarea1 is not None and bbinarea2 is not None:
            match_res_cor, match_res_err, remainin, remaintar = test_1frame(olareamodel, bbinarea1, bbinarea2, softmax, imgw, imgh, showid)
            if showid:
                for i in range(len(match_res_cor)):
                    label = 'bb id:%d' % (matchindex)
                    plot_one_box(bbinarea1_2[match_res_cor[i][0]][:4], im0, label=label, color=(255,0,0), person_count=0,pose_class='')
                    plot_one_box(bbinarea2_2[match_res_cor[i][1]][:4], im02, label=label, color=(255,0,0), person_count=0,pose_class='')
                    matchindex += 1
                for i in range(len(match_res_err)):
                    if len(bbinarea1_2) > match_res_err[i][0]:
                        label = 'bb id:%d' % (matchindex)
                        plot_one_box(bbinarea1_2[match_res_err[i][0]][:4], im0, label=label, color=(255,0,0), person_count=0,pose_class='')
                    if len(bbinarea2_2) > match_res_err[i][1]:
                        label = 'bb id:%d' % (matchindex+1)
                        plot_one_box(bbinarea2_2[match_res_err[i][1]][:4], im02, label=label, color=(255,0,0), person_count=0,pose_class='')
                    matchindex += 1
                if len(remainin)>0:
                    for i in remainin:
                        label = 'bb id:%d' % (matchindex)
                        plot_one_box(bbinarea1_2[i][:4], im0, label=label, color=(255,0,0), person_count=0,pose_class='')
                        matchindex += 1
                if len(remaintar)>0:
                    for i in remaintar:
                        label = 'bb id:%d' % (matchindex)
                        plot_one_box(bbinarea2_2[i][:4], im02, label=label, color=(255,0,0), person_count=0,pose_class='')
                        matchindex += 1
        if bbinarea1 is None: bbinarea1 = []
        if bbinarea2 is None: bbinarea2 = []
        if match_res_cor is None: match_res_cor = []
        remainbbinarea = len(bbinarea1) + len(bbinarea2) - len(match_res_cor)*2
        num_people = remainbbinarea + len(match_res_cor) + notinarea1 + notinarea2

        # Draw overlap area
        if drawRegion:
            cv2.line(im0, (0, thres_upbb_up), (imgw, thres_upbb_up), (0, 0, 255), 5)
            cv2.line(im0, (0, thres_upbb_down), (imgw, thres_upbb_down), (0, 0, 255), 5)
            cv2.line(im02, (0, thres_upbb_up), (imgw, thres_upbb_up), (0, 0, 255), 5)
            cv2.line(im02, (0, thres_upbb_down), (imgw, thres_upbb_down), (0, 0, 255), 5)
            cv2.line(im0, (0, thres_down), (imgw, thres_down), (0, 0, 255), 5)
            cv2.line(im0, (0, thres_down), (imgw, thres_down), (0, 0, 255), 5)
            cv2.line(im02, (0, thres_down), (imgw, thres_down), (0, 0, 255), 5)
            cv2.line(im02, (0, thres_down), (imgw, thres_down), (0, 0, 255), 5)
            cv2.line(im0, (chair_down, 0), (chair_down, imgh), (0, 0, 255), 5)
            cv2.line(im0, (chair_down, 0), (chair_down, imgh), (0, 0, 255), 5)
            cv2.line(im02, (imgw-chair_down, 0), (imgw-chair_down, imgh), (0, 0, 255), 5)
            cv2.line(im02, (imgw-chair_down, 0), (imgw-chair_down, imgh), (0, 0, 255), 5)
        tl_2 = round(0.002 * (im0.shape[0] + im0.shape[1]))
        tf = max(tl_2 - 1, 1)
        im0 = cv2.putText(im0, 'num='+str(num_people) , (50,50), 0, tl_2/3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)

        test_inf_time += (time.time()-t)
        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path1, im0)
                cv2.imwrite(save_path2, im02)
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
            width = 1280
            height = 1440
            if video_fir != video_sec:  # new video
                video_fir = video_sec
                if isinstance(vid_writer1, cv2.VideoWriter):
                    vid_writer1.release()  # release previous video writer
                fps = 10
                vid_writer1 = cv2.VideoWriter('images1.avi', cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            im0_re = cv2.resize(im0,(1280, 720))
            im02_re = cv2.resize(im02,(1280, 720))
            im = cv2.vconcat([im0_re, im02_re])
            vid_writer1.write(im)

        if num_people == 4: cor_count += 1
        else: errco.append(path1)

    print(errco)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output1)
        print('Results saved to %s' % os.getcwd() + os.sep + output2)

    # 910: in 1203
    # 266: in 1002
    #total_frame = 910
    total_frame = cor_count+len(errco)
    print('count score: %.3f(%d/%d)'%(cor_count/total_frame, cor_count, total_frame))
    print('inf time:', total_frame/test_inf_time)
    print('det time:', total_frame/test_det_time)
    print('match time:', total_frame/test_match_time)
    print('save time:', total_frame/test_actck_time)
    print('act time:', total_frame/test_act_time)
    print('overlap association:', total_frame/test_overlap_time)
    print(time.time()-t_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/obj.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best-detbest.pt', help='path to weights file')
    parser.add_argument('--images1', type=str, default='data/cam1-10-low', help='path to images')
    parser.add_argument('--images2', type=str, default='data/cam2-10-hf-low', help='path to images')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output1', type=str, default='output1', help='specifies the output path for images and videos')
    parser.add_argument('--output2', type=str, default='output2', help='specifies the output path for images and videos')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images1=opt.images1,
               images2=opt.images2,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output1=opt.output1,
               output2=opt.output2)
