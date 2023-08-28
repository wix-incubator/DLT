import os
import json
import torch
import pickle
import random
import numpy as np
from path import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from data_loaders.publaynet import PublaynetLayout
from data_loaders.data_utils import mask_whole_box, mask_loc, mask_size, mask_cat, mask_random_box_and_cat, mask_all
from utils import getDistinctColors


class RicoLayout(Dataset):
    # component_class = {'Text':0, 'Icon':1, 'Image':2, 'Text Button':3, 'Toolbar':4, 'List Item':5, 'Web View':6,
    # 'Advertisement':7, 'Input':8, 'Drawer':9, 'Background Image':10, 'Card':11, 'Multi-Tab':12, 'Modal':13,
    # 'Pager Indicator':14, 'Radio Button':15, 'On/Off Switch':16, 'Slider':17, 'Checkbox':18, 'Map View':19,
    # 'Button Bar':20, 'Video':21, 'Bottom Navigation':22, 'Date Picker':23, 'Number Stepper':24}
    component_class = {'Toolbar': 0, 'Image': 1, 'Text': 2, 'Icon': 3, 'Text Button': 4, 'Input': 5,
                       'List Item': 6, 'Advertisement': 7, 'Pager Indicator': 8, 'Web View': 9, 'Background Image': 10,
                       'Drawer': 11, 'Modal': 12}
    idx2class = {v + 1: k for k, v in component_class.items()}

    colors_f = getDistinctColors(len(component_class) + 2)
    name2color_map = {}
    idx2color_map = {}
    for i, c in enumerate(colors_f):
        if i == 0:
            name2color_map["empty_token"] = c
            idx2color_map[0] = c
        elif i == len(component_class) + 1:
            name2color_map["drop"] = c
            idx2color_map[len(component_class) + 1] = c
        else:
            name2color_map[idx2class[i]] = c
            idx2color_map[i] = c
    mask_func_map = {
        'whole_box': mask_whole_box,
        'loc': mask_loc,
        'size': mask_size,
        'cat': mask_cat,
        'random_box_and_cat': mask_random_box_and_cat,
        'all': mask_all
    }

    def __init__(self, data_path: Path, split: str = 'train', max_num_comp: int = 10, cond_type: str = None):
        # + 2 because 0 goes to pad and 1 goes to cat mask
        self.split = split
        self.categories_num = len(self.component_class.keys()) + 2
        self.data_path = data_path
        self.data = {
            'bbox': [],
            'file_idx': [],
            'annotations': [],
        }
        self.max_num_comp = max_num_comp
        self.process()
        self.cond_type = cond_type

    def get_all_element(self, p_dic, elements):
        if "children" in p_dic:
            for i in range(len(p_dic["children"])):
                cur_child = p_dic["children"][i]
                elements.append(cur_child)
                elements = self.get_all_element(cur_child, elements)
        return elements

    def __len__(self):
        return len(self.data['bbox'])

    def process(self):
        if (self.data_path / f'{self.split}.pth').exists():
            self.data = torch.load(self.data_path / f'{self.split}.pth')
            return
        else:
            data_dir = self.data_path / "semantic_annotations"
            bbox_idx = 0
            files = data_dir.files('*.json')
            for file in tqdm(files):
                with open(os.path.join(data_dir, file), "r") as f:
                    json_file = json.load(f)

                canvas = json_file["bounds"]

                W, H = float(canvas[2] - canvas[0]), float(canvas[3] - canvas[1])
                if canvas[0] != 0 or canvas[1] != 0 or W <= 1000:
                    continue
                elements = self.get_all_element(json_file, [])
                elements = list(filter(lambda e: e["componentLabel"] in self.component_class, elements))

                if len(elements) < 2 or len(elements) > self.max_num_comp:
                    continue

                ann_box = []
                ann_cat = []
                for ele in elements:
                    left, top, right, bottom = ele['bounds']
                    xc = (left + right) / 2.
                    yc = (top + bottom) / 2.
                    w = right - left
                    h = bottom - top

                    if w < 0 or h < 0:
                        continue
                    ann_box.append([xc, yc, w, h])
                    # componentLabel
                    ann_cat.append(
                        self.component_class[ele['componentLabel']])
                ann_box = np.array(ann_box)
                ann_cat = np.array(ann_cat)

                # we want to get order invariant model
                ind = [ttt for ttt in range(ann_box.shape[0])]
                random.shuffle(ind)
                ann_cat = ann_cat[ind]
                # shift class by 1 because 0 is for empty
                ann_cat += 1
                ann_box = ann_box[ind]

                ann_box = ann_box / np.array([W, H, W, H])
                ann_box = ((ann_box * 2) - 1) * 2
                self.data["bbox"].append(ann_box)
                self.data["file_idx"].append(file)
                self.data['annotations'].append(ann_cat)
                bbox_idx += 1
            # create train, val, test split indexes
            N = len(self.data['bbox'])
            s = [int(N * 0.85), int(N * 0.90)]
            torch.save({k: v[:s[0]] for k, v in self.data.items()}, self.data_path / "train.pth")
            torch.save({k: v[s[0]:s[1]] for k, v in self.data.items()}, self.data_path / "val.pth")
            torch.save({k: v[s[1]:] for k, v in self.data.items()}, self.data_path / "test.pth")
            self.data = torch.load(self.data_path / f'{self.split}.pth')

    def process_data_cond(self, idx, cond_type):
        box, cat, ind, name = self.get_data_by_ix(idx)

        mask_func = self.mask_func_map[cond_type]
        if mask_func == mask_random_box_and_cat:
            r_mask_box = np.random.uniform(0.5, 1.0, size=1)[0]
            r_mask_cat = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask_box=r_mask_box, r_mask_cat=r_mask_cat)
        elif mask_func == mask_all:
            mask, mask4cat = mask_func(box.shape)
        else:
            mask, mask4cat = mask_func(box.shape, r_mask=1.0)
        box, cat, mask, mask4cat = self.pad_instance(box, cat, mask, mask4cat, self.max_num_comp)
        return {
            "box": box.astype(np.float32),
            "cat": cat.astype(int),
            "box_cond": box.copy().astype(np.float32),
            "mask_box": mask.astype(int),
            "mask_cat": mask4cat.astype(int),
        }

    def get_data_by_ix(self, idx):
        box = self.data['bbox'][idx]
        ind = list(range(box.shape[0]))
        random.shuffle(ind)
        box = box[ind]
        cat = self.data['annotations'][idx][ind]

        name = self.data['file_idx'][idx]
        return box, cat, ind, name

    def process_data(self, idx):
        box, cat, ind, name = self.get_data_by_ix(idx)

        mask, mask4cat = self.mask_instance(box)

        box, cat, mask, mask4cat = self.pad_instance(box, cat, mask, mask4cat, self.max_num_comp)
        return {
            "box": box.astype(np.float32),
            "cat": cat.astype(int),
            "box_cond": box.copy().astype(np.float32),
            "mask_box": mask.astype(int),
            "mask_cat": mask4cat.astype(int),
        }

    @staticmethod
    def mask_instance(box):
        # here you can implement any masking strategy
        # mask, mask4cat = mask_all(box.shape)
        # mask_func = np.random.choice([mask_loc, mask_whole_box, mask_all], size=1,
        #                              p=[0.35, 0.35, 0.3])[0]
        # if mask_func == mask_all:
        #     mask, mask4cat = mask_func(box.shape)
        # else:
        #     r_mask = 1.0
        #     mask, mask4cat = mask_func(box.shape, r_mask=r_mask)
        mask_func = np.random.choice([mask_loc, mask_size, mask_cat, mask_whole_box, mask_random_box_and_cat,
                                      mask_all], 1,
                                     p=[0.2, 0.1, 0.05, 0.25, 0.1, 0.3])[0]
        if mask_func == mask_random_box_and_cat:
            r_mask_box = np.random.uniform(0.5, 1.0, size=1)[0]
            r_mask_cat = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask_box=r_mask_box, r_mask_cat=r_mask_cat)
        elif mask_func == mask_all:
            mask, mask4cat = mask_func(box.shape)
        else:
            r_mask = np.random.uniform(0.5, 1.0, size=1)[0]
            mask, mask4cat = mask_func(box.shape, r_mask=r_mask)
        return mask, mask4cat

    @staticmethod
    def pad_instance(box, cat, mask, mask4cat, max_num_comp=9):
        box = np.pad(box, pad_width=((0, max_num_comp - box.shape[0]), (0, 0)), constant_values=0.)
        cat = np.pad(cat, pad_width=(0, max_num_comp - cat.shape[0]), constant_values=0.)
        mask = np.pad(mask, pad_width=((0, max_num_comp - mask.shape[0]), (0, 0)), constant_values=0.)
        mask4cat = np.pad(mask4cat, pad_width=(0, max_num_comp - mask4cat.shape[0]),
                          constant_values=0.)
        return box, cat, mask, mask4cat

    def __getitem__(self, idx):
        if self.cond_type:
            assert self.cond_type in self.mask_func_map.keys()
            return self.process_data_cond(idx, self.cond_type)
        sample = self.process_data(idx)
        return sample
