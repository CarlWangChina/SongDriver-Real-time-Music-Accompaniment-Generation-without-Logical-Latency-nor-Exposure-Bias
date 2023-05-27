#pragma once
#include <fstream>
#include <tuple>
#include "chcpystring.h"
#include "generator.h"
#include "search.h"
namespace midiSearch {

using melody_t = std::vector<int>;
using chord_t = std::vector<std::vector<int>>;

struct music {
    melody_t melody;
    melody_t relativeMelody;
    chord_t chord;
    std::string name;
};

//构建相对值
template <typename relativeArray>
relativeArray buildRelativeArray(const relativeArray& arr) {
    relativeArray res;
    int last;
    bool first = true;
    for (auto it : arr) {
        if (it != 0) {
            if (!first) {
                auto delta = it - last;
                if (delta != 0) {
                    res.push_back(delta);
                }
            }
            last = it;
            first = false;
        }
    }
    return res;
}

//将相对值转为字符串
template <typename relativeArray>
std::string buildRelativeString(const relativeArray& arr) {
    std::ostringstream res;
    for (auto it : arr) {
        res << it << " ";
    }
    return res.str();
}

//逐行读取
inline generator<std::string*> lineReader(std::string path) {
    std::ifstream infile(path);
    if (!infile) {
        printf("fail to open:%s\n", path.c_str());
    }
    music res;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.size() > 1) {
            //切换协程
            co_yield &line;
        }
    }
}

inline void str2melody(const chcpy::string& str, melody_t& melody) {
    auto notes_arr = str.replace("[", "").replace("]", "").split(",");
    for (auto it : notes_arr) {
        int note = it.toInt();
        melody.push_back(note);
    }
}

inline void str2chord(const chcpy::string& ostr, chord_t& chords) {
    chcpy::string str = ostr;
    auto chord_str_array = str.mid(2, str.size() - 4).replace("],", "").split("[");
    for (auto it : chord_str_array) {
        chcpy::stringlist arr = it.split(",");
        std::vector<int> chord;
        for (auto n : arr) {
            int chord_note = n.toInt();
            if (chord_note > 0) {
                chord.push_back(chord_note);
            }
        }
        chords.push_back(std::move(chord));
    }
}

//3列的文件读取器（协程），返回值：music对象
inline generator<std::tuple<music, std::string*>> musicReader_3colume(std::string path) {
    music res;
    for (auto buf : lineReader(path)) {
        try {
            //开始切分字符串
            chcpy::string line = buf->c_str();
            auto line_array = line.simplified().split("|");
            auto name = line_array.at(0);
            auto notes_str = line_array.at(1);

            //旋律
            melody_t melody;
            str2melody(notes_str, melody);

            auto chord_str = line_array.at(2);
            //和弦
            chord_t chords;
            str2chord(chord_str, chords);

            res.name = name;
            res.melody = melody;
            res.chord = chords;
            res.relativeMelody = buildRelativeArray(res.melody);
            //切换协程
            co_yield std::make_tuple(res, buf);
        } catch (...) {
        }
    }
}

//2列的文件读取器（协程），返回值：旋律
inline generator<std::tuple<melody_t, chord_t>> musicReader_2colume(std::string path) {
    music res;
    for (auto buf : lineReader(path)) {
        try {
            //开始切分字符串
            chcpy::string line = buf->c_str();
            auto line_array = line.simplified().split("|");
            auto notes_str = line_array.at(0);
            auto chord_str = line_array.at(1);

            //旋律
            melody_t melody;
            str2melody(notes_str, melody);

            chord_t chords;
            str2chord(chord_str, chords);

            //切换协程
            co_yield std::tuple<melody_t, chord_t>(melody, chords);
        } catch (...) {
        }
    }
}

}  // namespace midiSearch