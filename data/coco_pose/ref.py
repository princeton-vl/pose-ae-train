import numpy as np
import pickle
import h5py
from scipy.misc import imread
import os 
from pycocotools.coco import COCO
from pycocotools import mask 

data_dir = 'coco/images'
ann_path = 'coco/annotations/person_keypoints_train2014.json'
ref_dir = os.path.dirname(__file__)

assert os.path.exists(data_dir)
assert os.path.exists(ann_path)
coco, img_ids, num_examples = None, None, None

with open(ref_dir + '/valid_id', 'r') as f:
    valid_id = list(map(lambda x:int(x.strip()), f.readlines()))
valid_id_set = set(valid_id)

def init():
    global coco, img_ids, num_examples
    ann_file = os.path.join(ann_path)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    num_examples = len(img_ids)

num_parts = 17
part_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1])
part_ref = {'ankle':[15,16],'knee':[13,14],'hip':[11,12],
            'wrist':[9,10],'elbow':[7,8],'shoulder':[5,6],
            'face':[0,1,2],'ears':[3,4]}
part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
basic_order = ['sho_l','sho_r', 'nose', 'eye_l','eye_r','ear_l',
               'ear_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
pairRef = [
    [1,2],[2,3],[1,3],
    [6,8],[8,10],[12,14],[14,16],
    [7,9],[9,11],[13,15],[15,17],
    [6,7],[12,13],[6,12],[7,13]
]
pairRef = np.array(pairRef) - 1

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

part_idx = {b:a for a, b in enumerate(part_labels)}
basic_order = [part_idx[i] for i in basic_order]


def initialize(opt):
    return

def image_path(idx):
    img_info = coco.loadImgs(img_ids[idx])[0]
    path = img_info['file_name'].split('_')[1] + '/' + img_info['file_name']
    return os.path.join(data_dir, path)

def load_image(idx):
    return imread(image_path(idx),mode='RGB')


def num_objects(idx, anns=None, should_greater_than_1 = False):
    if anns is None: anns = get_anns(idx)
    return len(anns)

def setup_val_split(opt = None):
    if coco is None:
        return [], []

    tmp_idxs = []
    for i in range(num_examples):
        if num_objects(i, None) > 0:
            tmp_idxs += [i]
    ref_idxs = np.array(tmp_idxs,dtype=int)
    ### choose image_id from valid_id_set

    valid = {}
    train = []
    for i in ref_idxs:
        if img_ids[i] in valid_id_set:
            valid[ img_ids[i] ]=i
        else:
            train.append(i)
    return np.array(train), np.array([valid[i] for i in valid_id if i in valid])

def get_anns(idx):
    ann_ids = coco.getAnnIds(imgIds=img_ids[idx])
    tmp_ann = coco.loadAnns(ann_ids)
    # Filter tmp_ann for people with no keypoints annotated
    return [tmp_ann[i] for i in range(len(tmp_ann)) if tmp_ann[i]['num_keypoints'] > 0]

def get_mask(idx):
    ann_ids = coco.getAnnIds(imgIds=img_ids[idx])
    anns = coco.loadAnns(ann_ids)
    img = coco.loadImgs(img_ids[idx])[0]
    m = np.zeros((img['height'], img['width']))
    for j in anns:
        if j['iscrowd']:
            rle = mask.frPyObjects(j['segmentation'], img['height'], img['width'])
            m += mask.decode(rle)
    return m < 0.5

def get_keypoints(idx, anns=None):
    if anns is None: anns = get_anns(idx)
    num_people = num_objects(idx, anns)
    kps = np.zeros((num_people, 17, 3))
    for i in range(num_people):
        kps[i] = np.array(anns[i]['keypoints']).reshape([-1,3])
    return kps
