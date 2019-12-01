from torch.utils.data import Dataset
import pickle
from PIL import Image
from PIL.ImageDraw import Draw
import torchvision.transforms as transforms
import os
from .datasets import pad_to_square, resize
import numpy as np
import torch
from utils.augmentations import horisontal_flip
import random

def _convert_box_cord(human_box, box):
    '''convert the origianl box with relative cord of human box,
       the box out of the human box will be cropped
    '''
    x1, y1, x2, y2 = box
    hx1, hy1, hx2, hy2 = human_box
    hw1 = hx2-hx1
    hh1 = hy2-hy1

    x1 = x1 - hx1
    y1 = y1 - hy1
    x2 = x2 - hx1
    y2 = y2 - hy1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(hw1-1, x2)
    y2 = min(hh1-1, y2)

    if x1 >= x2 or y1>= y2:
        return None
    else:
        return [x1, y1, x2, y2]

def _to_xyxy(box):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    return box


def draw_img(img, image):
    draw = Draw(img)

    for obj in image['objects']:
        draw.rectangle(obj['bbox'])
    img.save('test.jpg')


class ModanetHumanDataset(Dataset):
    def __init__(self, anno_root, image_root, part, image_size=416, use_revised_box=True, normalized_labels=False, augment=False, multiscale=False, box_extend=0.0):
        with open(anno_root, 'rb') as f:
            self.images = pickle.load(f)[part]
        self.im_root = image_root
        self.use_revised_box=use_revised_box
        self.img_size = image_size
        self.normalized_labels = normalized_labels
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.augment = augment
        self.box_extend = box_extend
        self._convert_human_box(extend=box_extend)
        # convert all the box cordinates to relative cordinate with human box
        self._convert_all_box_cord()
        self.batch_count = 0

    def __getitem__(self, idx):
        # ---------
        #  Image
        # ---------
        image = self.images[idx]
        im_path = os.path.join(self.im_root, image['file_name'])

        # Extract cropped image as PyTorch tensor
        box = image['human_box'] if self.use_revised_box else image['human_box_det']
        im = Image.open(im_path).convert('RGB')
        cropped_im = im.crop(box)
        
        img = transforms.ToTensor()(cropped_im)

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        targets = None
        #if os.path.exists(label_path):
        if len(image['objects'])>0:
            boxes = np.zeros((len(image['objects']), 5), dtype=float)
            for i, a_object in enumerate(image['objects']):
                boxes[i,0] = a_object['category_id'] - 1
                boxes[i, 1:] = a_object['bbox']
            boxes = torch.from_numpy(boxes)
            #boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] )
            y1 = h_factor * (boxes[:, 2] )
            x2 = w_factor * (boxes[:, 3] )
            y2 = h_factor * (boxes[:, 4] )
            #if x1<= 0 or x2>=w or y1<=0 or y2>=h:
            #    print('x1:{}, y1:{}, x2:{}, y2:{}')
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] = (x2-x1) * w_factor / padded_w #TODO check here later
            boxes[:, 4] = (y2-y1) * h_factor / padded_h

            targets = {}
            targets['origin'] = torch.zeros((len(boxes), 6))
            #targets = torch.zeros((len(boxes), 6))
            targets['origin'][:, 1:] = boxes
            targets['pad'] = pad
            targets['scale'] = max(h, w) / self.img_size
            targets['im_id'] = image['id']

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets['origin'] = horisontal_flip(img, targets['origin'])

        return im_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        if targets[0] is None:
            targets = None
        else:
            # Remove empty placeholder targets
            targets = [target for target in targets if target is not None]
            # Add sample index to targets
            for i, target in enumerate(targets):
                boxes = target['origin']
                boxes[:, 0] = i
            boxes = torch.cat(tuple([target['origin'] for target in targets]), 0)
            pads = [target['pad'] for target in targets]
            scales = [target['scale'] for target in targets]
            im_ids = [target['im_id'] for target in targets]

            targets = {}
            targets['origin'] = boxes
            targets['pad'] = pads
            targets['scale'] = scales
            targets['im_id'] = im_ids
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.images)

    def _convert_all_box_cord(self):
        ''' convert all the box to xyxy and convert the boxes to cord relative to human box,
            remove the boxes out of the human box
        '''
        for image in self.images:
            for obj in image['objects']:
                human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
                obj['bbox'] = _convert_box_cord(human_box, _to_xyxy(obj['bbox']))
                if obj['bbox'] is None:
                    image['objects'].remove(obj)

    def _convert_human_box(self, extend=0):
        '''Just make sure all the human box are inside the image'''
        for image in self.images:
            human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
            x1 = int(max(0, human_box[0]*(1-extend)))
            y1 = int(max(0, human_box[1]*(1-extend)))
            x2 = int(min(image['width']-1, human_box[2]*(1+extend)))
            y2 = int(min(image['height']-1, human_box[3]*(1+extend)))
            if self.use_revised_box:
                image['human_box'] = [x1, y1, x2, y2]
            else:
                image['human_box_det'] = [x1, y1, x2, y2]

