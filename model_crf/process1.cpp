#include <set>
#include "filereader.h"
#include "process.hpp"
int test1() {
    int index = 0;
    autochord::chordPredictor_t model;
    autochord::chordPredictor_loadModel(model, "./model.txt");
    auto fp = fopen("./output.txt", "w");
    double timeSum = 0;
    int timeNum = 0;
    for (auto line : midiSearch::musicReader_2colume("test.txt")) {
        auto& chord_ori = std::get<1>(line);
        auto& melody = std::get<0>(line);
        printf("第%d首\n", ++index);
        autochord::musicBuffer buffer;
        int note_index = 0;
        midiSearch::chord_t chords;
        chords.push_back(std::vector<int>());
        auto ch_it = chord_ori.begin();
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
                count.clear();
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
                printf("%d->%s %f\n",
                       max_note, outChord.c_str(),
                       std::get<1>(model.probOut.at(0)));
                if (ch_it == chord_ori.end()) {
                    buffer.pushChord(outChord);
                } else {
                    chcpy::stringlist sl;
                    for (auto it : *ch_it) {
                        sl.push_back(chcpy::string::number(it % 12));
                    }
                    auto ch_str = chcpy::join(sl, "-");
                    buffer.pushChord(ch_str);
                }

                if (outChord == "null") {
                    chords.push_back(std::vector<int>());
                } else {
                    auto arr = outChord.split("-");
                    int last = 36;
                    std::vector<int> schord;
                    for (auto it : arr) {
                        int n = it.toInt();
                        while (n <= last) {
                            n += 12;
                        }
                        last = n;
                        schord.push_back(n);
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
int test2() {
    int index = 0;
    autochord::chordPredictor_t model;
    autochord::chordPredictor_loadModel(model, "./model.txt");
    auto fp = fopen("./outputs/test.output.txt", "w");
    double timeSum = 0;
    int timeNum = 0;
    int escChordNum = 0;
    std::string escChord = "";
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
                count.clear();
                buffer.pushNote(max_note);

                clock_t startTime, endTime;
                startTime = clock();  //计时开始

                autochord::chordPredictor_getProb(model,
                                                  buffer.melody,
                                                  buffer.chord,
                                                  buffer.id);

                if (escChordNum > 0) {
                } else {
                    std::set<std::string> chordExist;
                    for (auto& it : buffer.chord_section) {
                        chordExist.insert(it);
                    }
                    if (chordExist.size() == 1) {
                        escChord = *chordExist.begin();
                        escChordNum = 4;
                    }
                }

                endTime = clock();  //计时结束
                timeSum += (double)(endTime - startTime) / CLOCKS_PER_SEC;
                ++timeNum;
                chcpy::string outChord;
                if (escChordNum > 0) {
                    for (auto it : model.probOut) {
                        outChord = std::get<0>(it);
                        if (outChord != escChord) {
                            break;
                        }
                    }
                    --escChordNum;
                } else {
                    outChord = std::get<0>(model.probOut.at(0));
                }
                printf("%d\t->\t%s\t%f\n",
                       max_note, outChord.c_str(),
                       std::get<1>(model.probOut.at(0)));
                buffer.pushChord(outChord);

                if (outChord == "null") {
                    chords.push_back(std::vector<int>());
                } else {
                    auto arr = outChord.split("-");
                    int last = 36;
                    std::vector<int> schord;
                    for (auto it : arr) {
                        int n = it.toInt();
                        while (n <= last) {
                            n += 12;
                        }
                        last = n;
                        schord.push_back(n);
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
    printf("用时%lf秒\n", timeSum);
    printf("平均用时%lf秒\n", timeSum / timeNum);
    if (fp) {
        fclose(fp);
    }
    return 0;
}

int main() {
    return test2();
}