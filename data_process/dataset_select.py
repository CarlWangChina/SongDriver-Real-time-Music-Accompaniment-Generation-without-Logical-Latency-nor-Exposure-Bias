import os
import sys
import random

cnt = 0
all = ''
for root, dir, files in os.walk('./dataset_old'):
    for file in files:
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(root, file), 'r', encoding='utf8') as f:
            all_lines = f.read().split('\n')
            for i in range(170):
                try:
                    i = random.randint(0, len(all_lines)-1)
                    print(all_lines[i])
                    name, melody, chord = all_lines[i].split('|')
                    melody = eval(melody)
                    chord = eval(chord)
                    start = random.randint(0, len(melody)-4*4-1)
                    sample_melody = melody[start:start+4*4].copy()
                    sample_chord = chord[start//4:start//4+4]
                    line = f'{str(sample_melody)}|{str(sample_chord)}\n'
                    all += line
                except Exception as e:
                    print(e)
        f.close()

with open('./dataset_random.txt', 'w') as f:
    f.write(all)
    f.close()

                # name, melody, chord = item.split('|')


print(cnt)
