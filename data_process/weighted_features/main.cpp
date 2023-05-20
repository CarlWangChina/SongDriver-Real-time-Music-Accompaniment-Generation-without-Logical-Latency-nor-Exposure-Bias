#include <iostream>
#include "chcpystring.h"
#include "melody2chord.h"
int main() {
    chcpy::melody2chord::chordMap chordmap;
    chcpy::melody2chord::musicSection_t section;
    chcpy::melody2chord::musicSection_t last;

    while (1) {
        chcpy::string buffer;
        std::cin >> buffer;
        auto arr = buffer.trimmed().split(",");
        std::vector<std::tuple<int, int> > melody;
        for (auto it : arr) {
            auto pair = it.split("/");
            if (pair.size() >= 2) {
                melody.push_back(std::tuple<int, int>(pair.at(0).toInt(), pair.at(0).toInt()));
            }
        }
        auto tmp = chcpy::melody2chord::buildMusicSection(chordmap, melody);
        if (section.melody.empty()) {
            section = tmp;
        } else {
            auto meg = chcpy::melody2chord::merge(chordmap, section, tmp);
            float nowWeight = section.weight + tmp.weight + 1.;
            if ((nowWeight >= meg.weight && meg.melody.size() < 64) || meg.melody.size() < 8) {
                section = std::move(meg);
            } else {
                last = section;
                section = std::move(tmp);
            }
        }
        chcpy::melody2chord::musicSection_t* target;
        if (section.melody.size() < 8) {
            target = &last;
        } else {
            target = &section;
        }
        auto it = chordmap.chord_map.find(target->chord_name);
        if (it == chordmap.chord_map.end()) {
            printf("%d\n", target->chord_base);
        } else {
            for (auto delta : it->second) {
                printf("%d ", target->chord_base + delta);
            }
            printf("\n");
        }
    }
    return 0;
}