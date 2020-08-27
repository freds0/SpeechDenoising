import os
import random
import argparse
import json
import torch
import torch.utils.data
from glob import glob
from tqdm import tqdm
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.generic_utils import load_config
#from multiprocessing import Process

def  create_spec(filepath, new_filepath):
    # extract spectrogram
    spectrogram, phase = ap.get_spec_from_audio_path(filepath)        
    # save spectrogram
    
    torch.save(spectrogram, new_filepath)
    return spectrogram, phase

if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--wavfile_path", default='output', required=False)
    parser.add_argument('-c', '--config', type=str, default='config.json', required=False,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    ap = AudioProcessor(config.audio)
    
    # Make directory if it doesn't exist
    '''
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)
    '''
    for filepath in tqdm(glob(args.wavfile_path + '/*.wav')):

        filename = os.path.basename(filepath)
        new_filepath = os.path.join(args.output_dir, filename.replace('.wav', '') + '.pt')
        #p = Process(target=create_spec, args=(filepath, new_filepath) )
        #p.start()
        create_spec(filepath, new_filepath)         
       
        # reverse spectrogram for wave file using Griffin-Lim
        #wav = ap.inv_spectrogram(spectrogram, phase)
        #ap.save_wav(wav, os.path.join(args.output_dir, filename + '.wav'))

    #p.join()
    #print("Spectogram with shape:",spectrogram.shape, "Saved in", new_filepath)