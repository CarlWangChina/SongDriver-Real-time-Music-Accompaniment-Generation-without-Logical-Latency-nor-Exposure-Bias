#pragma once
#include <coroutine>
#include <iostream>
#include <list>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "calcEditDist.h"
#include "generator.h"
#include "relative.h"
namespace midiSearch {

using melody_t = std::vector<int>;
using mu_melody_t = std::vector<melody_t>;
using chord_t = std::vector<std::vector<int> >;

struct music {
    melody_t melody;
    melody_t relativeMelody;
    chord_t chord;
    std::string name;
};

using musicMap = std::unordered_map<std::string, music>;
using musicList = std::list<std::tuple<std::vector<int>, std::vector<std::string> > >;

//构建音频字典
inline musicMap buildMusicMap(generator<std::tuple<music, char*> >& reader) {  //输入：读取器（协程）
    musicMap res;
    for (auto mu : reader) {
        res[buildRelativeString(std::get<0>(mu).relativeMelody)] = std::get<0>(mu);
    }
    return res;
}

//获取首调
template <typename melody_c>
inline int getFirstTone(const melody_c& melody) {
    for (auto it : melody) {
        if (it != 0) {
            return it;
        }
    }
    return 0;
}

}  // namespace midiSearch