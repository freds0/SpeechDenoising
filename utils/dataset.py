import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import stack
import numpy as np

class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """
    def __init__(self, c, ap, train=True):
        self.c = c
        self.ap = ap
        self.train = train
        self.dataset_dir = c.dataset['train_dir'] if train else c.dataset['test_dir']
        assert os.path.isdir(self.dataset_dir),'Test or Train dataset dir is incorrect! Fix it in config.json'
        
        format_data = c.dataset['format']
        self.target_spec_list = self.find_files_by_format(format_data['target'])
        self.mixed_spec_list = self.find_files_by_format(format_data['mixed'])
        self.target_wav_list = self.find_files_by_format(format_data['target_wav'])
        self.mixed_wav_list = self.find_files_by_format(format_data['mixed_wav'])
        # asserts for integrity
        assert len(self.target_spec_list) == len(self.mixed_spec_list), " The number of target and mixed Specs not Match! Check its"
        assert len(self.target_spec_list) != 0, " Training files not found !"

    def find_files_by_format(self, glob_exp):
            return sorted(glob(os.path.join(self.dataset_dir, glob_exp)))

    def __getitem__(self, idx):
        if self.train:
            mixed_wav = self.ap.load_wav(self.mixed_wav_list[idx])
            mixed_spec, mixed_phase = self.ap.get_spec_from_audio(mixed_wav, return_phase=True)
            target_wav = self.ap.load_wav(self.target_wav_list[idx])
            seq_len = torch.from_numpy(np.array([mixed_wav.shape[0]]))
            mixed_phase = torch.from_numpy(np.array(mixed_phase))
            mixed_spec = torch.from_numpy(mixed_spec)
            target_wav = torch.from_numpy(target_wav)
            return torch.load(self.target_spec_list[idx]), mixed_spec, seq_len, target_wav, mixed_phase  
        else: # if test
            # target_spec = torch.load(self.target_spec_list[idx])
            # mixed_spec = torch.load(self.mixed_spec_list[idx])
            mixed_wav = self.ap.load_wav(self.mixed_wav_list[idx])
            target_wav = self.ap.load_wav(self.target_wav_list[idx])
            mixed_spec, mixed_phase = self.ap.get_spec_from_audio(mixed_wav, return_phase=True)
            target_spec, _ = self.ap.get_spec_from_audio(target_wav, return_phase=True)
            target_spec = torch.from_numpy(target_spec)
            mixed_spec = torch.from_numpy(mixed_spec)
            mixed_phase = torch.from_numpy(mixed_phase)
            target_wav = torch.from_numpy(target_wav)
            mixed_wav = torch.from_numpy(mixed_wav)
            seq_len = torch.from_numpy(np.array([mixed_wav.shape[0]]))
            return target_spec, mixed_spec, target_wav, mixed_wav, mixed_phase, seq_len

    def __len__(self):
        return len(self.mixed_spec_list)
def train_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=True),
                          batch_size=c.train_config['batch_size'],
                          shuffle=True,
                          num_workers=c.train_config['num_workers'],
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

def test_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=False),
                          collate_fn=test_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])

def eval_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=False),
                          collate_fn=eval_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])


def eval_collate_fn(batch):
    return batch

def train_collate_fn(item):
    target_list = []
    mixed_list = []
    seq_len_list = []
    mixed_phase_list = []
    target_wav_list = []
    for target, mixed, seq_len, target_wav, mixed_phase in item:
        #target_list.append(target)
        target_list.append(torch.from_numpy(target))
        mixed_list.append(mixed)
        seq_len_list.append(seq_len)
        mixed_phase_list.append(mixed_phase)
        target_wav_list.append(target_wav)

    # concate tensors in dim 0
    #target_list = stack(target_list, dim=0)
    target_list = pad_sequence(target_list, batch_first=True, padding_value=0)

    #mixed_list = stack(mixed_list, dim=0)
    mixed_list = pad_sequence(mixed_list, batch_first=True, padding_value=0)

    seq_len_list = stack(seq_len_list, dim=0)
    
    #target_wav_list = stack(target_wav_list, dim=0)
    target_wav_list = pad_sequence(target_wav_list, batch_first=True, padding_value=0)

    #mixed_phase_list = stack(mixed_phase_list, dim=0) # np.array(mixed_phase_list)
    mixed_phase_list = pad_sequence(mixed_phase_list, batch_first=True, padding_value=0)

    return target_list, mixed_list, seq_len_list, target_wav_list, mixed_phase_list

def test_collate_fn(batch):
    target_list = []
    mixed_list = []
    seq_len_list = []
    mixed_phase_list = []
    target_wav_list = []
    mixed_wav_list = []
    
    for target, mixed, target_wav, mixed_wav, mixed_phase, seq_len in batch:
        target_list.append(target)
        mixed_list.append(mixed)
        seq_len_list.append(seq_len)
        mixed_phase_list.append(mixed_phase)
        target_wav_list.append(target_wav)
        mixed_wav_list.append(mixed_wav)

    # concate tensors in dim 0
    target_list = stack(target_list, dim=0)
    mixed_list = stack(mixed_list, dim=0)
    seq_len_list = stack(seq_len_list, dim=0)
    target_wav_list = stack(target_wav_list, dim=0)
    mixed_phase_list = stack(mixed_phase_list, dim=0) # np.array(mixed_phase_list)
    mixed_wav_list = stack(mixed_wav_list, dim=0)
    return target_list, mixed_list, target_wav_list, mixed_wav_list, mixed_phase_list, seq_len_list
  
