import os
from tqdm import tqdm
import json
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose

@DATASETS.register_module()
class PartDataset(Dataset):
    
    def __init__(
        self,
        category=None,
        data_root="./partnet/dataset",
        random_catogory=False,
        transform=None,
        loop=1,
        **kwargs
    ):
        super(PartDataset, self).__init__()
        assert os.path.exists(data_root), f"Data root {data_root} does not exist"
        assert random_catogory or category is not None, "Category must be specified when random_catogory is False"
        self.data_root = data_root
        self.random_catogory = random_catogory
        self.category = category
        self.data_list = []
        self.embedding = None
        self.transform = Compose(transform)
        self.prepare_data()
    
    def prepare_data(self):
        folders = [folder for folder in os.listdir(self.data_root) if (os.path.isdir(os.path.join(self.data_root, folder)) and int(folder) < 100000)]
        for folder in tqdm(folders, desc=f"Loading {len(folders)} folders", unit="folder"):
            folder_path = os.path.join(self.data_root, folder)
            
            with open(os.path.join(folder_path, 'result.json'), 'r', encoding='utf-8') as f:
                category = json.load(f)[0].get('name')

            if not self.random_catogory and category != self.category:
                continue

            self.data_list.append(folder)

        self.data_list = sorted(self.data_list)
        self.embedding = torch.from_numpy(
            np.load(os.path.join(self.data_root, f'../{self.category}.npy'))
        ).cuda().float()
        
    def get_data(self, idx):
        category = self.category
        folder = self.data_list[idx]
        
        folder_path = os.path.join(self.data_root, folder)
        label_file_path = os.path.join(folder_path, "point_sample/label-10000-new.txt")
        pcd_file_path = os.path.join(folder_path, "point_sample/sample-points-all-pts-nor-rgba-10000.ply")
        
        pc = o3d.io.read_point_cloud(pcd_file_path)
        point = np.asarray(pc.points) * 10
        rgba = np.asarray(pc.colors)
        normal = np.asarray(pc.normals)
        
        with open(label_file_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f]
        labels = torch.tensor(labels, dtype=torch.long)  
        
        pcd = {
              'coord': point,
              'color': rgba[...,:3],
              'normal': normal,
              'segment': labels.numpy(),
        }
        pcd = self.transform(pcd)
        
        batch = {
            'point': point,
            'color': rgba[...,:3],
            'pcd': pcd,
            'labels': labels,
            'category': category,
            'embedding': self.embedding,
            'obj_id': folder
        }
        
        return batch
    
    def val_data(self, training=True):
        if training:
            return self.__getitem__(np.random.randint(0, len(self)))
        else:
            return [self.get_data(i) for i in range(len(self))]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        ret = []
        n = len(self.data_list)
        if n > 8:
            id = np.random.choice(n, size=8, replace=False)
            for idx in id:
                obj = self.get_data(idx)
                ret.append(obj)
        else:
            for i in range(n):
                obj = self.get_data(i)
                ret.append(obj)
            
        return ret
    
if __name__ == "__main__":
    dataset = PartDataset(category="faucet", data_root="./partnet/dataset")
    data = dataset.get_data(0)
    print(type(data))
