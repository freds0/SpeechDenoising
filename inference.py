import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, load_config_from_str
from utils.generic_utils import set_init_dict

#from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

#from utils.generic_utils import validation, PowerLaw_Compressed_Loss, SiSNR_With_Pit

from models.voicefilter.model import VoiceFilter
from models.voicesplit.model import VoiceSplit
from models.voicedenoising.model import VoiceDenoising
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 

def inference(args, log_dir, checkpoint_path, testloader, c, model_name, ap, cuda=True):
    if(model_name == 'voicefilter'):
        model = VoiceFilter(c)
    elif(model_name == 'voicesplit'):
        model = VoiceSplit(c)
    elif(model_name == 'voicedenoising'):
        model = VoiceDenoising(c)
    else:
        raise Exception(" The model '"+model_name+"' is not suported")

    step = 0
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if cuda:
                model = model.cuda()
        except:
            raise Exception("Fail in load checkpoint, you need use this configs: %s" %checkpoint['config_str'])
        
    else:
        raise Exception("You need specific a checkpoint for inference")

    # convert model from cuda
    if cuda:
        model = model.cuda()

    model.eval()
    count = 0
    with torch.no_grad():
        for batch in testloader:
            try:
                mixed_spec, mixed_wav, mixed_phase, seq_len = batch[0]

                mixed_spec = mixed_spec.unsqueeze(0)

                if cuda:
                    mixed_spec = mixed_spec.cuda()

                est_mask = model(mixed_spec)
                est_mag = est_mask * mixed_spec
                mixed_spec = mixed_spec[0].cpu().detach().numpy()

                est_mag = est_mag[0].cpu().detach().numpy()
                mixed_phase = mixed_phase[0].cpu().detach().numpy()

                est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)
                est_mask = est_mask[0].cpu().detach().numpy()

                count+=1
                print(count, 'of',testloader.__len__())
            except:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str, default='./',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config_path', type=str, required=False, default=None,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    if args.config_path:
        c = load_config(args.config_path)
    else: #load config in checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        c = load_config_from_str(checkpoint['config_str'])

    ap = AudioProcessor(c.audio)

    audio_config = c.audio[c.audio['backend']]
    tensorboard = TensorboardWriter(log_path, audio_config)
    # set test dataset dir
    c.dataset['test_dir'] = args.dataset_dir

    test_dataloader = test_dataloader(c, ap)
    inference(args, log_path, args.checkpoint_path, test_dataloader, c, c.model_name, ap, cuda=True)