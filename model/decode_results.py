
import math
import os
import sys
import traceback
from tqdm import tqdm
from multiprocessing.pool import Pool
import subprocess
import numpy as np
import json

src_dict_path = './data/melody_to_id.json'
with open(src_dict_path) as f:
    src_dict = json.load(f)
melody_to_id = {k: v for k, v in src_dict.items()}
id2melody = {v: k for k, v in melody_to_id.items()}

def decode_result_job(src_path, save_dir):
    all_txt_lines = ''
    save_path = f'{save_dir}/{os.path.basename(src_path)}'
    with open(src_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            melody, chord = line.split('|')
            melody = eval(melody)
            new_melody = []
            for pitch_id in melody:
                new_melody.append(int(id2melody[pitch_id]))
            all_txt_lines += f'{str(new_melody)}|{chord}'
        f.close()
    with open(save_path, 'w') as f:
        f.write(all_txt_lines)
        f.close()


def decode_result(src_dir, dst_dir):
    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(decode_result_job, args=[
        os.path.join(src_dir, result), dst_dir,
    ]) for result in path_list if ".DS_Store" not in result]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()

if __name__ == "__main__":
    root_r = '/Users/leongtsihaw/Desktop/WAiMusic_copy/data/'
    root_d = '/Users/leongtsihaw/Desktop/WAiMusic_copy/data/'
    # dirs = [('no_cadence', 'SD-RC'), ('all_removed', 'SD_RA'), ('no_struct', 'SD_RS'), ('no_weighted', 'SD_RW'), ('no_isw', 'SD_RI')]
    dirs = [('results', 'results_decode')]
    for src_dir, dst_dir in dirs:
        decode_result(os.path.join(root_r, dst_dir), os.path.join(root_d, dst_dir))