import os
from tqdm import tqdm

def read_txt(pth):
        name, melody_str, chord_str = [], [], []
        with open(pth, 'r') as f:
                for i in f:
                        m, c = i.strip('\n').split('|')
                        melody_str.append(m)
                        chord_str.append(c)
        return melody_str, chord_str

def subdivide_dataset (src_path:str, dst_dir:str):
        melody_str, chord_str = read_txt(src_path)
        dst_name = os.path.basename(src_path)[:-4] + '_divided.txt'
        dst_path = os.path.join(dst_dir, dst_name)
        final_str = ''
        for s_idx, song in enumerate(tqdm(melody_str)):
                try:
                        song_notes = eval(song)
                        note_4set = []
                        c_idx = 0
                        final_str = ''
                        for idx in range (0, len(song_notes), 4):
                                note_4set = []
                                c_idx = 0
                                final_str = ''
                                note_4set.extend([song_notes[idx], song_notes[idx+1], song_notes[idx+2], song_notes[idx+3]])
                                corr_chord = eval(chord_str[s_idx])[idx//4]
                                final_str += str(note_4set) + '|' + str([corr_chord]) + '\n'
                                # print("final:str", final_str)
                                with open(dst_path, 'a') as ds:
                                        ds.write(final_str)
                                ds.close()
                except Exception as e:
                        print(e)
        return final_str

if __name__ == "__main__":
        src_path = './dataset_random.txt'
        dst_dir = './'
        subdivide_dataset(src_path, dst_dir)