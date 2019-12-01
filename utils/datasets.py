import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

def im_path_2_id(path):
    try:
        im_id = int(path[-16:-4])
    except ValueError:
        im_id = int(path[-10:-4])
    return im_id

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            #path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            path.replace("COCO/", "COCO/labels/").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        #print(label_path)

        targets = {}
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            #targets = {}
            targets['origin'] = torch.zeros((len(boxes), 6))
            #targets = torch.zeros((len(boxes), 6))
            targets['origin'][:, 1:] = boxes
        else:
            targets['origin'] = None
        targets['pad'] = pad
        targets['scale'] = max(h, w) / self.img_size
        targets['im_id'] = im_path_2_id(img_path)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets['origin'] = horisontal_flip(img, targets['origin'])

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        if targets[0] is None:
            targets = None
        else:
            # Remove empty placeholder targets
            targets_val = [target for target in targets if target['origin'] is not None]
            # Add sample index to targets
            if len(targets_val) > 0:
                for i, target in enumerate(targets_val):
                    boxes = target['origin']
                    boxes[:, 0] = i
                boxes = torch.cat(tuple([target['origin'] for target in targets_val]), 0)
            else:
                boxes = None
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
        return len(self.img_files)

class ModanetListDataset(Dataset):
    def __init__(self, anno_path, image_root, part, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        '''Modanet dataset, part = 'train' or 'val' '''
        with open(anno_path, "rb") as f:
            self.images = pickle.load(f)[part]

        self.im_root = image_root
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        image = self.images[index]
        img_path = os.path.join(self.im_root, image['file_name'])
        #img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        #label_path = self.label_files[index % len(self.img_files)].rstrip()
        #print(label_path)

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
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] -1 )
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] -1 )
            #if x1<= 0 or x2>=w or y1<=0 or y2>=h:
            #    print('x1:{}, y1:{}, x2:{}, y2:{}')
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            ## adjust x1, y1, x2, y2 for incorrect labels
            #x1.clamp(0, padded_w -1)
            #y1.clamp(0, padded_h - 1)
            #x2.clamp(1, padded_w)
            #y2.clamp(1, padded_h)

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w #TODO check here later
            boxes[:, 4] *= h_factor / padded_h
            #boxes[:, 1:] = torch.clamp(boxes[:, 1:], 0.0, 1.0 )
            #a = torch.ge(boxes[:,1:], 1)
            #b = torch.ge(-boxes[:,1:], 0)
            #print('bigger than 1:', a)
            #print('less than 0:', b)

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

        return img_path, img, targets

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