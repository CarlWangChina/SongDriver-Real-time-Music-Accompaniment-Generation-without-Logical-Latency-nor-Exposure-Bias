import sys
import json


def split_array(arr):
    return [arr[i:i+4] for i in range(0, len(arr), 4)]


def parse_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
        notes = []
        # print(len(data["notes"][0]),len(data["chords"][0]))

        for i in range(len(data["chords"][0])):
            line_notes = []
            for j in range(4):
                try:
                    n = str(data["notes"][0][i*4+j])
                except Exception:
                    n = "0"
                line_notes.append(n)
            line_notes_str = ",".join(line_notes)
            ch = data["chords"][0][i]

            resstr = f"[{line_notes_str}]|[{ch}]"
            print(resstr)


if __name__ == '__main__':
    file_path = sys.argv[1]
    parse_json_file(file_path)
