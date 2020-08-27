import argparse
from tqdm import tqdm
from os.path import join
from glob import glob
from random import randint, uniform, seed
from scipy.io.wavfile import read, write
import torchaudio
import numpy as np
import torch 

noise_min_amp = 0.3   
noise_max_amp = 0.8

def insert_noise_wav(wav, noise_wav):
    noise_wav_len = noise_wav.shape[1]
    wav_len = wav.shape[1]

    noise_start_slice = 0
    if noise_wav_len > wav_len:
        noise_start_slice = randint(0,noise_wav_len-(wav_len+1))
    else:
        new_noise_wav = torch.zeros(wav_len) 
        new_noise_wav[:noise_wav_len] = noise_wav
        noise_wav = new_noise_wav.reshape([1,-1])
 
    # sum two diferents noise
    noise_wav = noise_wav[:,noise_start_slice:noise_start_slice+wav_len]

    # get random max amp for noise
    #max_amp = random.uniform(noise_min_amp, noise_max_amp)
    #reduct_factor = max_amp/float(noise_wav.max().numpy())
    reduct_factor = uniform(noise_min_amp, noise_max_amp)
    noise_wav = noise_wav*reduct_factor
    new_wav = wav + noise_wav
    return new_wav

def input_noise_on_wavs_from_dir(dataset_dir, dataset_noises, noise_csv, output_dir, force):
 
    with open(noise_csv, "r") as f:
        noise_list = f.readlines()
        num_noise_files = len(noise_list)-1

    i = 0
    for filepath in tqdm(glob(dataset_dir + '/*-target.wav')):

        wav, sr = torchaudio.load(filepath)
        noise_filepath = noise_list[randint(0, num_noise_files)].strip()
        noise_wav, sr = torchaudio.load(join(dataset_noises, noise_filepath.strip()))
        new_wav = insert_noise_wav(wav, noise_wav)

        output_filepath = filepath.replace('-target.wav', '-mix.wav')
        if force:
            torchaudio.save(output_filepath, new_wav, sr)
        else:
            print('gen ' + output_filepath)
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./') 
    parser.add_argument('--dataset_dir', default='output', help='Name of csv file')   
    parser.add_argument('--dataset_noises', default='Dataset-Noises/', help='Name of csv file')   
    parser.add_argument('--noise_csv', default='total_noises.txt', help='Name of csv file')   
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()
    seed(a=42)

    dataset_noises = join(args.base_dir, args.dataset_noises)
    noise_csv = join(args.base_dir, args.dataset_noises, args.noise_csv)
    dataset_dir = join(args.base_dir, args.dataset_dir)
    output_dir = args.dataset_dir
    input_noise_on_wavs_from_dir(dataset_dir, dataset_noises, noise_csv, output_dir, args.force)

if __name__ == "__main__":
    main()




