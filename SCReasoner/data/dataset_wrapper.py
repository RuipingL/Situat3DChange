import torch
from torch.utils.data import Dataset

from .build import DATASETWRAPPER_REGISTRY
from .data_utils import pad_tensors


@DATASETWRAPPER_REGISTRY.register()
class LeoObjPadDatasetWrapper(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.max_obj_len = args.max_obj_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        # print(data_dict.keys())
        data_dict['obj_fts'] = pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len, pad=1.0).float()   # O, num_points, 6
        if 'obj_fts_scene' in data_dict.keys():
            data_dict['obj_fts_scene'] = pad_tensors(data_dict['obj_fts_scene'], lens=self.max_obj_len, pad=1.0).float()
            data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) | (torch.arange(self.max_obj_len)< len(data_dict['obj_locs_scene']))  # O
        else:
            data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))   # O
        data_dict['obj_locs']= pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len, pad=0.0).float()   # O, 6
        if 'obj_locs_scene' in data_dict.keys():
            data_dict['obj_locs_scene'] = pad_tensors(data_dict['obj_locs_scene'], lens=self.max_obj_len, pad=0.0).float()
        return data_dict
