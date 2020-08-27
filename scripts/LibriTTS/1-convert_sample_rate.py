import argparse
from os import makedirs, system
from os.path import join, exists
from tqdm import tqdm

number_bits = 16
encoding = "signed-integer"
number_channels = 1

def convert_sr_wavs(filepath, sr, output_dir, force):
        filename = filepath.split('/')[-1].replace('.wav', '') + '-target.wav'
        new_filepath = join(output_dir, filename)
        if force:
            system("sox %s -V0 -c %d -r %d -b %d -e %s %s"% (filepath, int(number_channels), int(sr), int(number_bits), encoding, new_filepath))
        else:
            print("sox %s -V0 -c %d -r %d -b %d -e %s %s"% (filepath, int(number_channels), int(sr), int(number_bits), encoding, new_filepath))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='LibriTTS/train-clean-100')
    parser.add_argument('--input_file', default='metadata.csv')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--sr', default='22050')
    parser.add_argument('--force', action='store_true', default=False)

    args = parser.parse_args()

    with open(join(args.base_dir, args.input_file)) as f:
        content_file = f.readlines()

    if not exists(args.output_dir) and (args.force):
        makedirs(args.output_dir)

    for line in tqdm(content_file):
        _, wavfile, _ = line.split('|')
        convert_sr_wavs(wavfile, args.sr, args.output_dir, args.force)

if __name__ == "__main__":
    main()
