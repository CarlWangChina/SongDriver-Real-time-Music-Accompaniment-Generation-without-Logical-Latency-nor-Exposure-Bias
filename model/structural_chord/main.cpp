#include <iostream>
#include "chcpystring.h"
#include "melody2chord.h"
bool tone_nh[] = {true, false, true, false, true, true, false, true, false, true, false, true};
bool tone_1245_M[] = {true, false, true, false, false, true, false, true, false, false, false, false};
bool tone_1245_m[] = {false, false, true, false, true, false, false, false, false, true, false, true};
int main() {
    chcpy::melody2chord::chordMap chordmap;
    chcpy::melody2chord::musicSection_t section;
    while (1) {
        chcpy::string buffer;
        std::cin >> buffer;
        auto arr = chcpy::string(buffer.substr(1)).split(",");
        std::vector<int> melody;
        for (auto it : arr) {
            melody.push_back(it.toInt());
        }
        if (melody.empty() || buffer.size() < 2) {
            printf("error\n");
            goto end;
        }
        if (buffer.at(0) == 'M') {
            if (!tone_1245_m[melody.at(0) % 12]) {
                printf("false\n");
                goto end;
            }
        } else if (buffer.at(0) == 'm') {
            if (!tone_1245_m[melody.at(0) % 12]) {
                printf("false\n");
                goto end;
            }
        } else {
            printf("error\n");
            goto end;
        }
        for (auto it : melody) {
            if (it != 0) {
                if (!tone_nh[it % 12]) {
                    printf("false\n");
                    goto end;
                }
            }
        }
        if (chcpy::melody2chord::inChordMap(chordmap, melody)) {
            printf("true\n");
        } else {
            printf("false\n");
        }
    end:
        continue;
    }
    return 0;
}