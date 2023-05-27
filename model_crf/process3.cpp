#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <functional>
#include <set>
#include "calcEditDist.h"
#include "filereader.h"
#include "process.hpp"
autochord::chordPredictor_t model;
void process(const char* in, const char* out, int step) {
    clock_t startTime, endTime;
    startTime = clock();  //计时开始
    int beatNum = 0;
    int index = 0;
    auto fp = fopen(out, "w");
    if (fp == nullptr) {
        printf("fail to open %s\n", out);
        return;
    }
    //auto fp_o = fopen(out_o, "w");
    //if (fp_o == nullptr) {
    //    printf("fail to open %s\n", out_o);
    //    return;
    //}
    midiSearch::melody_t melody;
    midiSearch::chord_t chord_ori;
    for (auto data : midiSearch::lineReader(in)) {
        //printf("line\n");
        chcpy::string d(*data);
        auto arr = d.trimmed().split("|");
        if (arr.size() >= 2) {
            midiSearch::melody_t m{}, c{};
            midiSearch::str2melody(arr.at(0), m);
            for (auto it : m) {
                melody.push_back(it);
            }
            if (arr.at(1) != "E") {
                midiSearch::str2melody(arr.at(1), c);
            }
            chord_ori.push_back(c);
        }
    }

    autochord::musicBuffer buffer;
    int note_index = 0;
    midiSearch::chord_t chords;
    //chords.push_back(std::vector<int>());
    auto ch_it = chord_ori.begin();
    std::map<int, int> count;
    midiSearch::melody_t lastChord;

    for (auto note : melody) {
        ++note_index;
        count[note] += 1;
        if (note_index % 4 == 0) {
            ++beatNum;
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

            autochord::chordPredictor_getProb(model,
                                              buffer.melody,
                                              buffer.chord,
                                              buffer.id);

            chcpy::string outChord;
            int minEditDist = 9999;
            for (int i = 0; i < step; ++i) {
                chcpy::string tmpoutChord = std::get<0>(model.probOut.at(i));
                auto arr = tmpoutChord.split("-");
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
                auto len = chcpy::calcEditDist(schord, lastChord);
                if (len < minEditDist) {
                    minEditDist = len;
                    outChord = tmpoutChord;
                }
            }
            if (ch_it == chord_ori.end()) {
                buffer.pushChord(outChord);
            } else {
                chcpy::stringlist sl;
                //fprintf(fp_o, "%d[", max_note);
                for (auto it : *ch_it) {
                    sl.push_back(chcpy::string::number(it % 12));
                    //fprintf(fp_o, "%d ", it);
                }
                auto ch_str = chcpy::join(sl, "-");
                buffer.pushChord(ch_str);
                //fprintf(fp_o, "]%s ", buffer.chord.rbegin()->c_str());
                //fprintf(fp_o, "|%s\n", ch_str.c_str());
                ++ch_it;
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
                lastChord = schord;
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
    if (fp) {
        fclose(fp);
    }
    endTime = clock();  //计时结束
    auto timeDelta = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("%d拍,耗时：%lf\n", beatNum, timeDelta);
    //if (fp_o) {
    //    fclose(fp_o);
    //}
}
void process_melody(const char* in, const char* out, int step) {
    clock_t startTime, endTime;
    startTime = clock();  //计时开始
    int beatNum = 0;
    int index = 0;
    auto fp = fopen(out, "w");
    if (fp == nullptr) {
        printf("fail to open %s\n", out);
        return;
    }
    //auto fp_o = fopen(out_o, "w");
    //if (fp_o == nullptr) {
    //    printf("fail to open %s\n", out_o);
    //    return;
    //}
    midiSearch::melody_t melody;
    for (auto data : midiSearch::lineReader(in)) {
        //printf("line\n");
        chcpy::string d(*data);
        auto arr = d.trimmed().split("|");
        if (arr.size() >= 2) {
            midiSearch::melody_t m{}, c{};
            midiSearch::str2melody(arr.at(0), m);
            for (auto it : m) {
                melody.push_back(it);
            }
            if (arr.at(1) != "E") {
                midiSearch::str2melody(arr.at(1), c);
            }
        }
    }

    autochord::musicBuffer buffer;
    int note_index = 0;
    midiSearch::chord_t chords;
    chords.push_back(std::vector<int>());
    std::map<int, int> count;
    midiSearch::melody_t lastChord;
    int escChordNum = 0;
    std::string escChord = "";

    for (auto note : melody) {
        ++note_index;
        count[note] += 1;
        if (note_index % 4 == 0) {
            ++beatNum;
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
            //timeSum += (double)(endTime - startTime) / CLOCKS_PER_SEC;
            //++timeNum;
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
            //printf("%d\t->\t%s\t%f\n",
            //       max_note, outChord.c_str(),
            //       std::get<1>(model.probOut.at(0)));
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
    if (fp) {
        fclose(fp);
    }
    endTime = clock();  //计时结束
    auto timeDelta = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("%d拍,耗时：%lf\n", beatNum, timeDelta);
    //if (fp_o) {
    //    fclose(fp_o);
    //}
}
int getdir(const char* pathname, const std::function<void(const char*)>& callback) {
    DIR* path = NULL;
    path = opendir(pathname);

    if (path == NULL) {
        perror("failed");
        exit(1);
    }
    struct dirent* ptr;  //目录结构体---属性：目录类型 d_type,  目录名称d_name
    char buf[1024] = {0};
    while ((ptr = readdir(path)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        }
        //如果是目录
        if (ptr->d_type == DT_DIR) {
            sprintf(buf, "%s/%s", pathname, ptr->d_name);
            printf("目录:%s\n", buf);
            getdir(buf, callback);
        }
        if (ptr->d_type == DT_REG) {
            sprintf(buf, "%s/%s", pathname, ptr->d_name);  //把pathname和文件名拼接后放进缓冲字符数组
            int len = strlen(buf);
            if (len > 5) {
                if (buf[len - 1] == 't' &&
                    buf[len - 2] == 'x' &&
                    buf[len - 3] == 't') {
                    printf("文件:%s\n", buf);
                    callback(buf);
                }
            }
        }
    }
    return 0;
}
int main() {
    autochord::chordPredictor_loadModel(model, "./model.txt");
    /*
    //for (int j = 2; j <= 5; ++j) {
    for (int i = 0; i <= 30; i += 1) {
        char buf_in[128];
        char buf_out[128];
        char buf_out_o[128];
        snprintf(buf_in, sizeof(buf_in), "./outputs/results_decode/test_ckpt%d.txt", i);
        printf("%s\n", buf_in);
        snprintf(buf_out, sizeof(buf_out), "./outputs/results_decode-out/test_ckpt%d.crf.txt", i);
        process(buf_in, buf_out, 1);
    }
    //}
    process("./outputs/tf/40M.txt", "./outputs/tf/40M.out.txt", "");
    process("./outputs/tf/20M.txt", "./outputs/tf/20M.out.txt", "");
    */
    //process("./outputs/test_ckpt360.txt", "./outputs/test_ckpt360.out.txt", 5);
    //process("./outputs/tf300-390/test_ckpt340.txt", "./outputs/test_ckpt340.txt", 1);
    /*
    if (fork()) {
        getdir("./outputs/results(1)/results/all", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.same";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp1.crf", 1);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp1.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdri", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.same";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp2.crf", 1);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp2.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdrs", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.same";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp3.crf", 1);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp3.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdrwf", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.same";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp4.crf", 1);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp4.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else {
        getdir("./outputs/results(1)/results/sdrwn", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.same";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp5.crf", 1);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp5.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
    }

    if (fork()) {
        getdir("./outputs/results(1)/results/all", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.find";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp6.crf", 2);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line.py /tmp/tmp6.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdri", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.find";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp7.crf", 2);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line.py /tmp/tmp7.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdrs", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.find";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp8.crf", 2);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line.py /tmp/tmp8.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else if (fork()) {
        getdir("./outputs/results(1)/results/sdrwf", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.find";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp9.crf", 2);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line.py /tmp/tmp9.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
        exit(0);
    }else {
        getdir("./outputs/results(1)/results/sdrwn", [](const char* pathname) {
            auto out = std::string(pathname) + ".crf.find";
            auto tfp = fopen(out.c_str(), "r");
            if (tfp != 0) {
                printf("%s已存在\n", out.c_str());
                fclose(tfp);
                return;
            }
            process(pathname, "/tmp/tmp10.crf", 2);
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "python3 2line.py /tmp/tmp10.crf '%s'", out.c_str());
            printf("cmd:%s\n", cmd);
            system(cmd);
        });
    }
    */
    getdir("./outputs/results(1)", [](const char* pathname) {
        auto out = std::string(pathname) + ".crf.same";
        auto tfp = fopen(out.c_str(), "r");
        if (tfp != 0) {
            printf("%s已存在\n", out.c_str());
            fclose(tfp);
            return;
        }
        process_melody(pathname, "/tmp/tmp1.crf", 1);
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "python3 2line-same.py /tmp/tmp1.crf '%s'", out.c_str());
        printf("cmd:%s\n", cmd);
        system(cmd);
    });
    return 0;
}
