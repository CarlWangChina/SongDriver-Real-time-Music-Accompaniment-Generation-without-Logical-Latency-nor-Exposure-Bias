import math
import os
import sys

f = open(sys.argv[1])
o = open(sys.argv[2], "w")

line = f.readline()
while line:
    pair = line.replace(" ", "").split("|")
    if len(pair) >= 2:
        chord = pair[1].lstrip("[\r\n").rstrip("]\r\n").split("],[")
        melody = pair[0].strip("[]").split(",")
        l_m = len(melody)
        l_c = len(chord)
        print("音符:"+str(l_m)+" 和弦:"+str(l_c))
        for i in range(l_c):
            m_arr = []
            for j in range(4):
                try:
                    m_arr.append(melody[i*4+j])
                except Exception:
                    m_arr.append("0")
            index = int(math.floor(i/2)*2)
            o.write("["+(",".join(m_arr))+"]|["+chord[index]+"]\n")
    line = f.readline()
