import os
from tqdm import tqdm

def read_txt(pth):
        name, melody_str, chord_str = [], [], []
        with open(pth, 'r') as f:
                for i in f:
                        n, m, c = i.strip('\n').split('|')
                        name.append(n)
                        melody_str.append(m)
                        chord_str.append(c)
        return name, melody_str, chord_str

def subdivide_dataset (src_path:str, dst_dir:str):
        name, melody_str, chord_str = read_txt(src_path)
        dst_name = os.path.basename(src_path)[:-4] + '_divided.txt'
        dst_path = os.path.join(dst_dir, dst_name)
        final_str = ''
        for s_idx, song in enumerate(tqdm(melody_str)):
                try:
                        name_s = name[s_idx].replace('/', '_')
                        with open(f'/{name_s}.txt') as f:
                                a = f.readlines()
                        keys = []
                        for a_i in a:
                                keys.append(eval(a_i))
                        time = 0
                        key = 0
                        song_notes = eval(song)
                        note_4set = []
                        c_idx = 0
                        final_str = ''
                        for idx in range (0, len(song_notes), 4):
                                if key+1 < len(keys) and time > keys[key+1][0]:
                                        key += 1
                                root, quality = keys[key][-1].split(':KeyMode')
                                key_name = root + quality
                                note_4set = []
                                c_idx = 0
                                final_str = ''
                                note_4set.extend([song_notes[idx], song_notes[idx+1], song_notes[idx+2], song_notes[idx+3]])
                                corr_chord = eval(chord_str[s_idx])[idx//4]
                                final_str += str(note_4set) + '|' + key_name + '|' + str([corr_chord]) + '\n'
                                # print("final:str", final_str)
                                with open(dst_path, 'a') as ds:
                                        ds.write(final_str)
                                ds.close()
                                time += 0.25
                except Exception as e:
                        print(e)
        return final_str

if __name__ == "__main__":
        src_path = '/Users/leongtsihaw/Desktop/NextLab/PhraseLevelSegmentation/dataset/Dataset_AIGE.txt'
        dst_dir = '/Users/leongtsihaw/Desktop/WAiMusic/dataset'
        subdivide_dataset(src_path, dst_dir)