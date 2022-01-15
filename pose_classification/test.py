import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=512,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device()
        verbose = False

        # Initialize model
        model = DarknetRes(cfg, img_size).to(device)
        model_info(model, report='full')
        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.model1.load_state_dict(torch.load(weights, map_location=device)['model1'])
            model.model2.load_state_dict(torch.load(weights, map_location=device)['model2'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = True

    # Configure run
    data = parse_data_cfg(data)
    print(data)
    nc = int(data['classes'])  # number of classes

    test_path = data['valid']  # path to test images
    print(test_path)
    names = load_classes(data['names'])  # class names

    criterion = nn.CrossEntropyLoss()

    # Dataloader
    dataset2 = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader2 = DataLoader(dataset2,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset2.collate_fn)
    #print(dataset)
    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%30s' + '%10s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1', 'Acc')
    loss1, loss2, cla_acc, cla_total, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    n_sit, n_stand, correct_sit, correct_stand = 0., 0., 0., 0.
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader2, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        targets2 = targets.clone()
        targets[:,1] = 0
        '''
        print('\ntargets4yolo:')
        print(targets4yolo[:,1])
        print('\ntargets2:')
        print(targets2[:,1])
        '''

        if not len(targets)==0:
            # Run model
            # print(model)
            inf_out, train_out, cla_res = model(imgs, targets)  # inference and training outputs

            #print(type(train_out))
            #print('\ntargets3:')
            #print(targets2)
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss1 += compute_loss(train_out, targets, model)[0].item()

            loss_cla = criterion(cla_res, targets2[:, 1].to(device=device, dtype=torch.int64))
            loss2 += loss_cla.item()

            # compute accuracy of classifier
            _, predicted = cla_res.max(1)
            cla_total += targets2[:,1].to(device=device, dtype=torch.int64).size(0)
            cla_acc += predicted.eq(targets2[:,1].to(device=device, dtype=torch.int64)).sum().item()
            # cal classifier accuracy
            for c in range(targets2[:,1].to(device=device, dtype=torch.int64).size(0)):
                if targets2[c,1]==0:
                    if predicted[c].eq(targets2[c,1].to(device=device, dtype=torch.int64))==1:
                        correct_sit = correct_sit+1
                        n_sit = n_sit+1
                    else:
                        n_sit = n_sit+1
                if targets2[c,1]==1:
                    if predicted[c].eq(targets2[c,1].to(device=device, dtype=torch.int64))==1:
                        correct_stand = correct_stand+1
                        n_stand = n_stand+1
                    else:
                        n_stand = n_stand+1
            
            '''
            print('\ntargets:')
            print(targets[:,1])
            print('\ntargets2:')
            print(targets2[:,2:])
            print('\npred:')
            print(predicted)
            '''
            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
            #print("PP:")
            #print(np.shape(output[0]))
            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1
                if pred is None:
                    if nl:
                        stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Append to text file
            
                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(Path(paths[si]).stem.split('_')[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for di, d in enumerate(pred):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(d[6])],
                                      'bbox': [floatn(x, 3) for x in box[di]],
                                      'score': floatn(d[4], 5)})

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height

                    # Search for correct predictions
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        # Best iou, index between pred and targets
                        m = (pcls == tcls_tensor).nonzero().view(-1)
                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        # If iou > threshold and class is correct mark as correct
                        if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    cla_avg_acc = (correct_sit/n_sit + correct_stand/n_stand)/2

    # Print results
    pf = '%30s' + '%10.3g' * 7  # print format
    pf2 = '%30s' + '%10s' * 1 + '%10.3g' * 1 + '%10s' * 4 + '%10.3g' * 1  # print format for classifier
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1, cla_avg_acc))

    # Print results per class
    #changed
    #if verbose and nc > 1 and len(stats):
    classes = ['sit','stand']
    cla_n = [n_sit, n_stand]
    cla_list = [correct_sit/n_sit, correct_stand/n_stand]

    if nc > 1 and len(stats):
        for i in range(len(classes)):
            print(pf2 % (classes[i], '-', cla_n[i], '-', '-', '-', '-', cla_list[i]))
            #print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i], 0.0))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return (mp, mr, map, mf1, loss1 / len(dataloader2), loss2 / len(dataloader2), cla_avg_acc), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/obj.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
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
