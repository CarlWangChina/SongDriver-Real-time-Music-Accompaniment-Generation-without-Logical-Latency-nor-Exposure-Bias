#include "filereader.h"
#include "melody2chord.h"
#include "process.hpp"
int main() {
    int index = 0;
    autochord::chordPredictor_t model;
    autochord::chordPredictor_loadModel(model, "./model2.txt");
    auto fp = fopen("./output2.txt", "w");
    chcpy::melody2chord::chordMap chordmap;
    double timeSum = 0;
    int timeNum = 0;
    for (auto line : midiSearch::lineReader("./outputs/test.txt")) {
        std::vector<int> melody;
        midiSearch::str2melody(*line, melody);
        printf("第%d首\n", ++index);
        autochord::musicBuffer buffer;
        int note_index = 0;
        midiSearch::chord_t chords;
        chords.push_back(std::vector<int>());
        std::map<int, int> count;
        for (auto note : melody) {
            ++note_index;
            count[note] += 1;
            if (note_index % 4 == 0) {
                buffer.id = note_index / 16;
                int max_note_num = 0;
                int max_note = 0;
                for (auto it : count) {
                    if (it.second > max_note_num) {
                        max_note_num = it.second;
                        max_note = it.first;
                    }
                }
                buffer.pushNote(max_note);

                clock_t startTime, endTime;
                startTime = clock();  //计时开始

                autochord::chordPredictor_getProb(model,
                                                  buffer.melody,
                                                  buffer.chord,
                                                  buffer.id);

                endTime = clock();  //计时结束
                timeSum += (double)(endTime - startTime) / CLOCKS_PER_SEC;
                ++timeNum;
                chcpy::string outChord = std::get<0>(model.probOut.at(0));
                printf("%d\t->\t%s\t%f\n",
                       max_note, outChord.c_str(),
                       std::get<1>(model.probOut.at(0)));
                buffer.pushChord(outChord);

                if (outChord == "null") {
                    chords.push_back(std::vector<int>());
                } else {
                    auto arr = outChord.split("_");
                    //重建数组
                    int base = chcpy::string(arr.at(0)).toInt() + 36;
                    std::vector<int> schord;
                    auto chmit = chordmap.chord_map.find(arr.at(1));
                    if (chmit != chordmap.chord_map.end()) {
                        for (auto delta : chmit->second) {
                            schord.push_back(base + delta);
                        }
                    } else {
                        schord.push_back(base);
                    }
                    chords.push_back(schord);
                }
            }
        }
        if (fp) {
            chcpy::stringlist melody_strlist, chords_strlist;
            for (auto& it : melody) {
                melody_strlist.push_back(chcpy::string::number(it));
            }
            fprintf(fp, "[%s]|[", chcpy::join(melody_strlist, ",").c_str());
            for (auto& ch : chords) {
                chcpy::stringlist chord_list;
                for (auto& it : ch) {
                    chord_list.push_back(chcpy::string::number(it));
                }
                chords_strlist.push_back(
                    chcpy::string("[") +
                    chcpy::join(chord_list, ",") +
                    chcpy::string("]"));
            }
            fprintf(fp, "%s]\n", chcpy::join(chords_strlist, ",").c_str());
        }
    }
    printf("平均用时%lf秒\n", timeSum / timeNum);
    if (fp) {
        fclose(fp);
    }
    return 0;
}