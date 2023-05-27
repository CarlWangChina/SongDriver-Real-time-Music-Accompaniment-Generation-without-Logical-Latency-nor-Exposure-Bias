import miditoolkit
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import os
import sys
import pandas as pd
import miditoolkit
import traceback
from tqdm import tqdm
from multiprocessing.pool import Pool
import subprocess
import numpy as np
import qz_ss

# 一共七级和弦
chord_series = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
chord_name_major = ['I', '0', 'II', '0', 'III',
                    'IV', '0', 'V', '0', 'VI', '0', 'VII']
chord_name_minor = ['III', '0', 'IV', '0',
                    'V', 'VI', '0', 'VII', '0', 'I', '0', 'II']


class Chord:
    def __init__(self, name, start, end):
        self.name = name.strip().upper()
        self.start = start
        self.end = end
        self.dur = end-start

    def inDur(self, time):
        return time >= self.start and time <= self.end

    def print(self):
        print(
            f"Name:{self.name}   Start:{self.start}\tEnd:{self.end}\tDur:{self.dur}")


def group_by_interval(midi_path):
    if isinstance(midi_path, str):
        midi = miditoolkit.MidiFile(midi_path)
    else:
        midi = midi_path

    midi_group = {}
    for inst in midi.instruments:
        if inst.name == 'Kontakt Portable x64 01':
            inst.name = 'Melody'
        elif inst.name == 'Kontakt Portable x64 01 (D)':
            inst.name = 'Chord'
        if inst.name.upper() == 'LEAD' or inst.name.upper().strip() == 'MELODY':  # 只分析主旋律轨道
            group_id = 0
            for idx, note in enumerate(inst.notes):  # 收集音符
                if group_id not in midi_group.keys():
                    midi_group[group_id] = []
                if idx == 0 or (idx > 0 and note.start - inst.notes[idx-1].end < midi.ticks_per_beat):
                    midi_group[group_id].append(note)
                elif note.start - inst.notes[idx-1].end >= midi.ticks_per_beat:  # 至少2拍
                    group_id += 1
                    midi_group[group_id] = []
                    midi_group[group_id].append(note)
                else:
                    continue
        else:
            continue

    break_points = []
    for group_id, group in midi_group.items():
        break_points.append(group[-1].end)

    return midi_group, break_points


def group_by_ends(midi_path):
    if isinstance(midi_path, str):
        midi = miditoolkit.MidiFile(midi_path)
    else:
        midi = midi_path

    end_points = []

    for inst in midi.instruments:
        if inst.name == 'Kontakt Portable x64 01':
            inst.name = 'Melody'
        elif inst.name == 'Kontakt Portable x64 01 (D)':
            inst.name = 'Chord'
        if inst.name.upper() == 'LEAD' or inst.name.upper().strip() == 'MELODY':  # 只分析主旋律轨道
            group_id = 0
            for idx, note in enumerate(inst.notes):  # 收集音符
                end_points.append(note.end)

    return end_points

# 读取MIDI文件 转换为(ticks, pitch)的表征方式


def midi2tuple(midi_path):
    if isinstance(midi_path, str):
        midi = miditoolkit.MidiFile(midi_path)
    else:
        midi = midi_path

    time = []
    pitch = []

    for inst in midi.instruments:
        if inst.name == 'Kontakt Portable x64 01':
            inst.name = 'Melody'
        elif inst.name == 'Kontakt Portable x64 01 (D)':
            inst.name = 'Chord'
        if inst.name.upper() == 'LEAD' or inst.name.upper().strip() == 'MELODY':  # 只分析主旋律轨道
            for note in inst.notes:  # 收集音符
                time.append(note.start)
                pitch.append(note.pitch)
    data = [time, pitch]
    return data


number2note = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: "VII"}

# 读取MIDI文件 转换为(ticks, pitch)的表征方式


def midi2chords(midi_path):
    if isinstance(midi_path, str):
        midi = miditoolkit.MidiFile(midi_path)
    else:
        midi = midi_path
    chords = []

    if len(midi.markers) == 0:
        return None

    for idx, marker in enumerate(midi.markers):
        if marker.text.strip().upper() not in chord_series:  # Invalid Chord Information
            continue
        if idx < len(midi.markers)-1:
            chord = Chord(marker.text, marker.time, midi.markers[idx+1].time)
        elif idx == len(midi.markers)-1:
            chord = Chord(marker.text, marker.time,
                          midi.max_tick)  # Last Chord
        if chord.dur > 0:
            chords.append(chord)

    return chords


def mergeChords(chords):
    merged_chords = []
    for idx, chord in enumerate(chords):
        if len(merged_chords) == 0:
            merged_chords.append(chord)
            continue
        if chord.name == merged_chords[-1].name:
            continue
        else:
            merged_chords.append(chord)
    return merged_chords


def findBreakPoint(cand, *ref):
    m = {}
    for r in ref:
        m[r] = abs(np.min(r-np.array(cand)))
    m = sorted(m.items(), key=lambda x: x[1])
    minr = m[0][0]
    return cand[np.argmin(minr-np.array(cand))]


def findAdjacentBreak(location, endpoints):
    distance = abs(np.array(endpoints)-location)
    idx = np.argmin(distance)
    # print(len(distance), len(endpoints))
    # print("idx", idx)
    return endpoints[idx]


def detectCadence(chords, break_points):
    # 初始化终止式字典
    cadence = {}
    cadence_type = ['PC', 'PLC', 'IC', 'SC']
    for t in cadence_type:
        if t not in cadence.keys():
            cadence[t] = []

    # 第一类：正格终止(Perfect Cadence) 即V->I的和弦进行
    sub_chords = ['II', 'IV', 'VI']  # 正格终止前一般使用下属功能组和弦
    pefc = []  # 正格中止
    for i in range(1, len(chords)-1):
        if chords[i].name == 'V' and chords[i+1].name == 'I' and \
                chords[i-1].name in sub_chords:
            # 判断旋律切断点是否落在和弦时域内
            b = findAdjacentBreak(chords[i+1].end, break_points)
            pefc.append(b)
            cadence['PC'].append(b)

    # 第二类：变格终止(Plagal Cadence) 即IV->I的和弦进行
    plac = []
    for i in range(0, len(chords)-1):  # 变格终止
        if (chords[i].name == 'IV' or chords[i].name == 'II' or chords[i].name == 'VI') \
                and chords[i+1].name == 'I':  # 从V进行到I
            # print(chords[i].name, chords[i+1].name)
            # 判断旋律切断点是否落在和弦时域内
            b = findAdjacentBreak(chords[i+1].end, break_points)
            plac.append(b)
            cadence['PLC'].append(b)

    # 第三类：阻碍终止(Interrputed Cadencce) V7->I 被 V7->VI代替
    intc = []
    for i in range(0, len(chords)-3):  # 阻碍终止
        if (chords[i].name == 'V' and chords[i+1].name == 'I') and \
                (chords[i+2].name == 'V' and chords[i+3].name == 'VI'):  # 从V进行到I
            # 判断旋律切断点是否落在和弦时域内
            b = findAdjacentBreak(chords[i+3].end, break_points)
            intc.append(b)
            cadence['IC'].append(b)

    # 第四类：半终止(Semi Cadence) 任意和弦到V或VII（配合斜率）
    semc = []
    for i in range(1, len(chords)-1):
        # 从1开始 因为V与VII需要出现在其他和弦后面
        if chords[i].name == 'V' or chords[i].name == 'VII':
            # 判断旋律切断点是否落在和弦时域内
            b = findAdjacentBreak(chords[i].end, break_points)
            semc.append(b)
            cadence['SC'].append(b)

    return cadence


def detectCadence_arr(chords):

    # 第一类：正格终止(Perfect Cadence) 即V->I的和弦进行
    sub_chords = ['II', 'IV', 'VI']  # 正格终止前一般使用下属功能组和弦
    pefc = []  # 正格中止
    for i in range(1, len(chords)-1):
        if chords[-1] == 'I' and chords[-2] == 'V' and (chords[-3] in sub_chords):
            return True

    # 第二类：变格终止(Plagal Cadence) 即IV->I的和弦进行
    plac = []
    for i in range(0, len(chords)-1):  # 变格终止
        if (chords[-2] == 'IV' or chords[-2] == 'II' or chords[-2] == 'VI') and chords[-1] == 'I':  # 从V进行到I
            return True

    # 第三类：阻碍终止(Interrputed Cadencce) V7->I 被 V7->VI代替
    intc = []
    for i in range(0, len(chords)-3):  # 阻碍终止
        if (chords[-4] == 'V' and chords[-3] == 'I') and (chords[-2] == 'V' and chords[-1] == 'VI'):  # 从V进行到I
            return True

    # 第四类：半终止(Semi Cadence) 任意和弦到V或VII（配合斜率）
    semc = []
    for i in range(1, len(chords)-1):
        # 从1开始 因为V与VII需要出现在其他和弦后面
        if chords[-1] == 'V' or chords[-1] == 'VII':
            return True

    return False


def markCadence(cadence, midi_path, save_dir):
    m = miditoolkit.MidiFile(midi_path)
    markers = []
    for name, starts in cadence.items():
        # print(starts)
        for start in starts:
            n = miditoolkit.Marker(text=name, time=start)
            markers.append(n)
    m.markers = markers.copy()
    save_path = f'{save_dir}/{os.path.basename(midi_path)}'
    m.dump(save_path)


def group_by_measures(src_path):
    midi = miditoolkit.MidiFile(src_path)
    return [i for i in range(0, midi.max_tick, midi.ticks_per_beat * 4)]


def phraseLevelSegmentation_single(src_path, save_dir):
    chords_unmerged = midi2chords(src_path)
    chords = mergeChords(chords_unmerged)
    end_points = group_by_measures(src_path)
    cadence = detectCadence(chords, end_points)
    print(cadence)
    markCadence(cadence, src_path, save_dir)


def phraseLevelSegmentation(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)
    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(phraseLevelSegmentation_single, args=[
        os.path.join(src_dir, midi_fn), dst_dir,
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


if __name__ == "__main__":
    src_dir = '/home/ubuntu/project/ml/2022-3/2022-3-16ypf任务/2022-3-12-四个乐理对应的代码+1个流式处理+调性数据集[等下拉回去]/【终止和弦】采样格式下原始乐句分割代码-未删除音符层面/PhraseLevelSegmentation/annoteted_midi_series'
    dst_dir = '/home/ubuntu/project/ml/2022-3/2022-3-16ypf任务/2022-3-12-四个乐理对应的代码+1个流式处理+调性数据集[等下拉回去]/【终止和弦】采样格式下原始乐句分割代码-未删除音符层面/PhraseLevelSegmentation/midi_with_cadence'
    #src_dir = sys.argv[1]
    #dst_dir = sys.argv[2]
    phraseLevelSegmentation(src_dir, dst_dir)
