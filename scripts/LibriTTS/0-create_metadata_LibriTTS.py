import argparse
import os
from os.path import join, basename
from glob import glob
import re
import sys
import pandas as pd
from tqdm import tqdm

def create_metadata_libri_tts(root_path, outfile):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    for meta_file in tqdm(meta_files):
        _meta_file = basename(meta_file).split('.')[0]
        speaker_name = _meta_file.split('_')[0]
        chapter_id = _meta_file.split('_')[1]
        _root_path = join(root_path, f"{speaker_name}/{chapter_id}")
        with open(meta_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('\t')
                wav_file = join(_root_path, cols[0] + '.wav')
                text = cols[1]
                items.append([text, wav_file, speaker_name])

    if outfile is not None:
        df = pd.DataFrame(items, columns=['text','wav_file','speaker_name'])
        df.to_csv(outfile, index=False, sep='|')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='LibriTTS/train-clean-100')
    parser.add_argument('--output_file', default='metadata.csv')

    args = parser.parse_args()

    create_metadata_libri_tts(args.base_dir,  outfile=join(args.base_dir, args.output_file))


if __name__ == "__main__":
    main()