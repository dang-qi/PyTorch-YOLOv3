from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from PIL.ImageDraw import Draw

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.modanet_human_dataset import ModanetHumanDataset


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        targets = targets['origin']

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def extract_imgs(im_tensor):
    images = []
    for im in im_tensor:
        image = transforms.ToPILImage()(im).convert("RGB")
        images.append(image)
    return images

def draw_single_image(image, boxes): 
    # (x1, y1, x2, y2, object_conf, class_score, class_pred)
    if boxes is not None:
        draw = Draw(image)
        for box in boxes:
            if type(box) == list:
                return image
            if type(box) is not np.ndarray:
                box = box.detach().cpu().numpy()
            box_rec = box[:4]
            object_conf = box[4]
            class_score = box[5]
            class_pred = box[6]
            draw.rectangle(box_rec)
            draw.text((box[0], box[1]), '{:.2f} {}'.format(object_conf*class_score, class_pred))
    return image

def to_xywh(box):
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]
    return box

def draw_and_save(images, batch_boxes, path, ind_start):
    for i, (image, boxes) in enumerate(zip(images, batch_boxes)):
        image = draw_single_image(image, boxes)
        im_path = os.path.join(path, '{:06d}.jpg'.format(ind_start+i))
        image.save(im_path)

def get_max_score_box(boxes):
    if boxes == []:
        return []
    score = boxes[:,4]*boxes[:,5]
    ind = np.argmax(score)
    return boxes[ind]

# mainly for generate pickle file
def reconstruct_boxes(batch_boxes, pads, scales):
    reconstructed_boxes = []
    for pad, scale, boxes in zip(pads, scales, batch_boxes):
        if boxes is None:
            reconstructed_boxes.append([])
            continue
        boxes = boxes.detach().cpu().numpy()
        #bboxes = boxes[:,:4]
        #obj_score = boxes[:,4]
        #cat_score = boxes[:,5]
        #category = boxes[:,6]
        boxes[:,:4] = boxes[:,:4]*scale
        boxes[:, 0] = np.maximum(0, boxes[:,0]-pad[0])
        boxes[:, 2] = np.maximum(0, boxes[:,2]-pad[0])
        boxes[:, 1] = boxes[:,1]-pad[2]
        boxes[:, 3] = boxes[:,3]-pad[2]
        reconstructed_boxes.append(boxes)
    return reconstructed_boxes

        
# mainly for generate json file
def get_result_within_batch(pads, scales, im_ids, outputs):
    results=[]
    for pad, scale, im_id, boxes in zip(pads, scales, im_ids, outputs):
        if boxes is None:
            continue
        boxes = [box.detach().cpu().tolist() for box in boxes]
        scores = [box[4] for box in boxes]
        categorys = [box[6] for box in boxes]
        boxes = [np.array(box[:4]) for box in boxes]
        boxes = [(box*scale).tolist() for box in boxes]
        for category, score, box in zip(categorys, scores, boxes):
            box[0] = max(0, box[0] - pad[0])
            box[2] = max(0, box[2] - pad[0])
            box[1] = box[1] - pad[2]
            box[3] = box[3] - pad[2]
            #convert to xywh
            box = to_xywh(box)
            results.append({'image_id': im_id,
                            'category_id': int(category)+1,
                            'bbox': box,
                            'score': score })
    return results

def eval_result():
    gt_json='/ssd/data/datasets/COCO/annotations/instances_val2014.json'
    dt_json='temp_result.json'
    annType = 'bbox'
    cocoGt=COCO(gt_json)
    cocoDt=cocoGt.loadRes(dt_json)

    imgIds=sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.catIds = [1]
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def test_speed(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, out_path=None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    results = []
    start = time.time()
    im_num = 0
    for batch_i, ( _, imgs, targets) in enumerate(dataloader):
        im_num+=len(imgs)
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            '''
            outputs:
            list([(x1, y1, x2, y2, object_conf, class_score, class_pred)...]...)
            '''
    time_per_im = (time.time()-start) / im_num
    print('time per image is {}'.format(time_per_im))
    print('image number is {}'.format(im_num))
    
#def test_coco(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, out_path=None):
def test(model, anno_path, image_root, iou_thres, conf_thres, nms_thres, img_size, batch_size, out_path=None):
    model.eval()

    # Get dataloader
    #dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    #dataset = ModanetListDataset(anno_path, image_root, part='val', img_size=416)
    dataset = ModanetHumanDataset(anno_path, image_root, 'val', image_size=416)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    results = []
    #for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    im_num = 0
    for batch_i, ( paths, imgs, targets) in enumerate(dataloader):
        images = extract_imgs(imgs)
        #images = []
        #for path in paths:
        #    images.append(Image.open(path).convert('RGB'))

        pads = targets['pad']
        scales = targets['scale']
        im_ids = targets['im_id']
        im_num+=len(imgs)

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            '''
            outputs:
            list([(x1, y1, x2, y2, object_conf, class_score, class_pred)...]...)
            '''
            #batch_boxes = reconstruct_boxes(outputs, pads, scales)
            
            #draw_boxes = []
            #for boxes, im_id in zip(batch_boxes, im_ids):
            #    #box = get_max_score_box(boxes)
            #    #results.append({'image_id':im_id,
            #    #                 'human_box':box})
            #    #draw_boxes.append([box])
            #    draw_boxes.append(boxes)
        #ind_start = batch_i * batch_size + 1
        #draw_and_save(images, draw_boxes, out_path, ind_start)

    #with open('human_info_modanet.pkl', 'wb') as f:
    #    pickle.dump(results, f)
        #batch_results = get_result_within_batch(pads, scales, im_ids, outputs)

        ind_start = batch_i * batch_size + 1
        draw_and_save(images, outputs, out_path, ind_start)
    #with open('temp_result.json','w') as f:
    #    json.dump(results,f)
    #eval_result()
        #if batch_i == 20:
        #    break

def evaluate_modanet_human(model, anno_path, image_root, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    #dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    #dataset = ModanetListDataset(anno_path, image_root, part='val', img_size=416)
    dataset = ModanetHumanDataset(anno_path, image_root, 'val', image_size=img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        targets = targets['origin']

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
def evaluate_modanet(model, anno_path, image_root, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    #dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataset = ModanetListDataset(anno_path, image_root, part='val', img_size=416)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        targets = targets['origin']

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    #out_path = '/hdd/fashion_videos/yolo_frames'
    #out_path = '/hdd/data/datasets/modanet/person_detection_result'
    out_path = '/hdd/data/datasets/modanet/detection_result'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    if 'anno_path' in data_config and 'image_root' in data_config:
        anno_path = data_config['anno_path']
        image_root = data_config['image_root']

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    test(
        model,
        #path=valid_path,
        anno_path=anno_path,
        image_root=image_root,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        out_path=out_path
    )

    #test_coco(
    #    model,
    #    path=valid_path,
    #    #anno_path=anno_path,
    #    #image_root=image_root,
    #    iou_thres=opt.iou_thres,
    #    conf_thres=opt.conf_thres,
    #    nms_thres=opt.nms_thres,
    #    img_size=opt.img_size,
    #    batch_size=opt.batch_size,
    #    out_path=out_path
    #)
    #print("Compute mAP...")

    #precision, recall, AP, f1, ap_class = evaluate(
    #    model,
    #    path=valid_path,
    #    iou_thres=opt.iou_thres,
    #    conf_thres=opt.conf_thres,
    #    nms_thres=opt.nms_thres,
    #    img_size=opt.img_size,
    #    batch_size=8,
    #)

    #print("Average Precisions:")
    #for i, c in enumerate(ap_class):
    #    print("+ Class '{}' ({}) - AP: {}".format(c, class_names[c], AP[i]))

    #print("mAP: {}".format(AP.mean()))
