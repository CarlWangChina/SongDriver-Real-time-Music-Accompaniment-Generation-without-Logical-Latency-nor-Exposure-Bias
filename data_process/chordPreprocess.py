import json
import os
import string
import subprocess
from subprocess import Popen
import model
import qz_ss
import phrase_level_segmentation

src_dict_path = './data/melody_to_id.json'
with open(src_dict_path) as f:
    src_dict = json.load(f)
melody_to_id = {k: v for k, v in src_dict.items()}
id_to_melody = {v: k for k, v in melody_to_id.items()}
# print(melody_to_id)


tgt_dict_path = './data/chord_to_id.json'
with open(tgt_dict_path) as f:
    tgt_dict = json.load(f)
f.close()
chord_to_id = {k: v for k, v in tgt_dict.items()}
id_to_chord = {v: k for k, v in chord_to_id.items()}
# print(chord_to_id)

basetone = 0    # 1的音高 pitch CMajor [0-11]
isMajor = True  # 是否是大调


def solution(s):
    # 创建存放最终结果的列表
    a = []
    # 判断字符串个数情况
    b = len(s)
    if b >= 2:
        if b % 2 == 0:
            for i in range(0, b, 2):
                a.append(s[i:i+2])
            return a
        else:
            ss = s+'_'
            for i in range(0, b+1, 2):
                a.append(ss[i:i+2])
            return a
    else:
        if b == 1:
            ss = s+'_'
            for i in range(0, b+1, 2):
                a.append(ss[i:i+2])
            return a
        else:
            return a

class structural_chord:
    def __init__(self):
        self.p = subprocess.Popen("./structural_chord/main", shell=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def check(self, arr, chord_str):
        # print(arr, chord_str, model.structural_chord)
        buf = []
        for it in arr:
            buf.append(str(it))
        argstr = ""
        global isMajor
        if isMajor:
            argstr = "M"
        else:
            argstr = "m"
        argstr += (",".join(buf)+"\n")
        # print(argstr)
        self.p.stdin.write(argstr.encode("utf-8"))
        self.p.stdin.flush()
        res = (self.p.stdout.readline().decode("utf-8").strip()) == "true"
        if res:
            arr_id = [int(chord_to_id[chord_str])]
            # print(arr_id)
            model.structural_chord = arr_id.copy()
            # print("M:", model.structural_chord)
        return res


class weighted_features:
    def __init__(self):
        self.p = subprocess.Popen("./weighted_features/main", shell=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def get(self, arr):
        # print("arr", arr)
        buf = []
        for it in arr:
            buf.append(str(it))
        wstr = (",".join(buf)+"\n")
        # print(wstr)
        self.p.stdin.write(wstr.encode("utf-8"))
        self.p.stdin.flush()
        resbuf = (self.p.stdout.readline().decode("utf-8").strip()).split()
        res = ''
        if len(resbuf) == 1:
            model.weighted_features = [int(chord_to_id[res])].copy()
            # print(res)
            return res
        for it in resbuf:
            n = int(it)
            if n != 0:
                n += basetone
            try:
                res += str(n)
            except Exception:
                res += str("00")
        model.weighted_features = res
        # model.weighted_features = [int(chord_to_id[res])].copy()
        # print("res:", res)
        return res


class qz_chord:
    def __init__(self):
        self.sp = qz_ss.StreamProcessor()

    def get(self, arr):
        self.sp.pushSeq(arr)
        try:
            qz_list = qz_ss.get_qz(self.sp.buffer.copy(),
                                   self.sp.lastSection[-1])
        except Exception as e:
            print(e)
        model.is_weighted = []
        arr_w = qz_list[-4:]
        # print("arr_w", arr_w)
        arr_f = []
        for it in arr_w:
            arr_f.append(it.strip("()").replace(",", "/"))
        # print("arr_f:", arr_f)
        weighted_features_checker.get(arr_f)
        # print(arr_f)
        for it in arr_w:
            pair = str(it).split(",")
            # print(pair)
            if len(pair) >= 2 and '2' in pair[1]:
                model.is_weighted.append(1)
            else:
                model.is_weighted.append(0)
        return False


chords = qz_ss.StreamProcessor()

structural_chord_checker = structural_chord()
weighted_features_checker = weighted_features()
qz_chord_checker = qz_chord()


def pushMelody(arr):
    tmp = []
    for it in arr:
        n = int(it)
        if n != 0:
            n = n - basetone
        tmp.append(n)
    qz_chord_checker.get(tmp)
    # print(model.weighted_features)


def pushChord(str):
    arr_str = solution(str)
    if len(arr_str) > 0:
        arr = []
        for it in arr_str:
            if int(it) != 0:
                arr.append(int(it)-basetone)
            else:
                arr.append(0)
        structural_chord_checker.check(arr, str)

        if isMajor:
            chords.push(
                phrase_level_segmentation.chord_name_major[int(arr[0]) % 12])
        else:
            chords.push(
                phrase_level_segmentation.chord_name_minor[int(arr[0]) % 12])
        if (phrase_level_segmentation.detectCadence_arr(chords.buffer)):
            model.is_cadence = 1
        else:
            model.is_cadence = 0
    # print(model.structural_chord)


if __name__ == '__main__':
    # print(structural_chord_checker.check([12, 22, 32, 42]))
    # print(weighted_features_checker.get([1, 2, 3, 4]))
    # print(qz_chord_checker.get([36, 36, 36, 36]))
    # print(qz_chord_checker.get([36, 36, 36, 36]))
    # print(qz_chord_checker.get([36, 36, 36, 36]))
    # print(qz_chord_checker.get([36, 36, 36, 36]))
    # print(qz_chord_checker.get([36, 36, 36, 36]))
    pass
