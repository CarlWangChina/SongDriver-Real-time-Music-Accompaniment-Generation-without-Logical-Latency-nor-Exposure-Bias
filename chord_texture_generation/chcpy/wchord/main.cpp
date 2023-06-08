#include <iostream>
#include "chcpystring.h"
#include "melody2chord.h"
int main() {
    chcpy::melody2chord::chordMap chordmap;
    chcpy::melody2chord::musicSection_t section;
    while (1) {
        chcpy::string buffer;
        std::cin >> buffer;
        auto arr = buffer.split(",");
        std::vector<int> melody;
        for (auto it : arr) {
            melody.push_back(it.toInt());
        }
        auto tmp = chcpy::melody2chord::buildMusicSection(chordmap, melody);
        if (section.melody.empty()) {
            section = tmp;
        } else {
            auto meg = chcpy::melody2chord::merge(chordmap, section, tmp);
            float nowWeight = section.weight + tmp.weight + 1.;
            if (nowWeight >= meg.weight && meg.melody.size() < 32) {
                section = std::move(meg);
            } else {
                section = std::move(tmp);
            }
        }
        auto it = chordmap.chord_map.find(section.chord_name);
        if (it == chordmap.chord_map.end()) {
            printf("%d\n", section.chord_base);
        } else {
            for (auto delta : it->second) {
                printf("%d ", section.chord_base + delta);
            }
            printf("\n");
        }
    }
    return 0;
}