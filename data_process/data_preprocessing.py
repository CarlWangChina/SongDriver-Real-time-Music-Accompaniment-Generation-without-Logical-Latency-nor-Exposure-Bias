import json
import numpy as np
from collections import defaultdict


def read_txt(pth):
    """
    :param pth: the path of the dataset which is saved as a txt file.
    :return:
    """
    melody_str, key_str, chord_str, weighted_factor, weighted_notes, structural_chord, is_cadence = [], [], [], [], [], [], []
    with open(pth, 'r') as f:
        for i in f.readlines():
            m, k, c, wf, wn, sc, ic = i.strip('\n').split('|')
            key_str.append(k)
            melody_str.append(m)
            chord_str.append(c)
            weighted_factor.append(wf)
            weighted_notes.append(wn)
            structural_chord.append(sc)
            is_cadence.append(ic)
    return key_str, melody_str, chord_str, weighted_factor, weighted_notes, structural_chord, is_cadence

def flag_str_to_list(flag_str):
    flag_lst = []
    for i in flag_str:
        flag_lst.append(eval(i))
    return flag_lst

def melody_str_to_list(melody_str):
    melody_lst = []
    for i in melody_str:
        full_str = i.lstrip('[').rstrip(']').split(', ')
        full_str = ['0' + i if len(i) == 1 else i for i in full_str]
        sentence = []
        # print(f'full_str:{full_str}')
        for ele in [''.join(full_str[i:i + 1]) for i in range(0, len(full_str), 1)]:
            # print(f"ele:{ele}")
            sentence.append(ele)
        melody_lst.append(sentence)
    # print(f'melody_list: {melody_lst}')
    return melody_lst


def chord_str_to_list(chord_str):
    chord_lst = []
    for i in chord_str:
        full_str = i.lstrip('[[').rstrip(']]').split('], [')
        sentence = []
        for ele in full_str:
            ele = ele.lstrip('[').rstrip(']').split(', ')
            ele = ['0' + i if len(i) == 1 else i for i in ele]
            sentence.append(''.join(ele))
        chord_lst.append(sentence)
    return chord_lst


def make_dict(seq_lst):
    unique_lst = []
    for i in seq_lst:
        unique_lst.extend(i)
    unique_lst = np.unique(np.sort(unique_lst))
    seq_to_id = {ele: i + 1 for i, ele in enumerate(unique_lst)}
    # seq_to_id = sorted(seq_to_id.keys())
    return seq_to_id



if __name__ == '__main__':
    key_str, melody_str, chord_str, wf, _, wn, _ = read_txt('dataset/all-half.pre.full.txt')

    melody_lst = melody_str_to_list(melody_str)

    wf_list = melody_str_to_list(wf)
    wn_list = melody_str_to_list(wn)

    chord_lst = chord_str_to_list(chord_str)

    import pickle
    key2chord = pickle.load(open('./key2chord.pkl', 'rb'))
    for k,v in key2chord.items():
        std_chord = ''
        for note in v:
            std_chord += str(note)
        if std_chord == '798486':
            print("YES")
        chord_lst.append([std_chord])

    for ch in wf_list:
        std_chord = ''
        for note in ch:
            std_chord += str(note)
        chord_lst.append([std_chord])

    for ch in wn_list:
        std_chord = ''
        for note in ch:
            std_chord += str(note)
        chord_lst.append([std_chord])

    chord_lst.append(['000000'])
    melody_to_id = make_dict(melody_lst)
    chord_to_id = make_dict(chord_lst)
    melody_to_id['P'] = 0
    melody_to_id['S'] = len(melody_to_id)
    melody_to_id['E'] = len(melody_to_id)
    chord_to_id['P'] = 0
    chord_to_id['S'] = len(chord_to_id)
    chord_to_id['E'] = len(chord_to_id)

    # melody_to_id = dict(sorted(melody_to_id.items(), key = lambda x:x[0]))
    # chord_to_id = dict(sorted(chord_to_id.items(), key = lambda x:x[0]))

    with open('data/melody_to_id.json', 'w') as f:
        json.dump(melody_to_id, f)
    with open('data/chord_to_id.json', 'w') as f:
        json.dump(chord_to_id, f)
