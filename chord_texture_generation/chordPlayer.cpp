#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <functional>
#include "MidiFile.h"
#include "Options.h"
#include "chordScript.hpp"
#include "filereader.h"
#include "sectioner.hpp"

struct chordScript : autochord::chordScript_t, autochord::sectioner {
    int bpm, bps, fragId, baseTone;
    float averDelta;
    std::vector<int> nowChord;
};

#define E3 40
#define C4 48
void buildMidi(const char* out, const char* sample) {
    int longChord = 0;
    int chordCount = 0;
    smf::MidiFile midifile;
    midifile.setTPQ(384 / 4);  //0音轨
    midifile.addTrack();
    midifile.addTempo(0, 0, 120);
    midifile.addTrack();
    midifile.addTimbre(0, 0, 0, 0);
    midifile.addTimbre(1, 0, 0, 0);
    midifile.addTrack();
    midifile.addTimbre(2, 0, 0, 0);
    midifile.addTrack();
    midifile.addTimbre(3, 0, 0, 24);
    midifile.addTrack();
    midifile.addTimbre(4, 0, 0, 95);
    midifile.addTrack();

    int tm = 0;

    int lastNote = 0;
    int lastNoteStart = 0;

    chordScript script;
    script.scriptPath = "./default.lua";
    autochord::chordScript_init(script);
    autochord::sectioner_init(script, 16);
    script.fragId = 0;
    script.bps = 4;
    script.baseTone = 0;
    script.playNote_available = true;
    int nowTm = 0;

    script.playNote = [&](int tone, int vel, int channel) {
        if (vel > 0) {
            //printf("%d %d %d\n", tone, vel, channel);
            //if (channel == 1) {
            //    midifile.addNoteOn(2, nowTm, 0, tone, vel);
            //} else if (channel == 2) {
            //    midifile.addNoteOn(3, nowTm, 0, tone, vel);
            //}
            //if (channel == 1) {
            //    printf("%d %d %d\n", tone, vel, channel);
            //}
            midifile.addNoteOn(channel + 1, nowTm, channel + 1, tone, vel);
        } else {
            //if (channel == 1) {
            //    midifile.addNoteOff(2, nowTm, 0, tone, vel);
            //} else if (channel == 2) {
            //    midifile.addNoteOff(3, nowTm, 0, tone, vel);
            //}
            midifile.addNoteOff(channel + 1, nowTm, channel + 1, tone, vel);
        }
    };
    script.setIns = [&](int a, int b) {
    };

    for (auto data : midiSearch::lineReader(sample)) {
        chcpy::string d(*data);
        auto arr = d.trimmed().split("|");
        if (arr.size() >= 2) {
            midiSearch::melody_t ch, chord_list, note_list;
            midiSearch::str2melody(arr.at(0), note_list);
            midiSearch::str2melody(arr.at(1), chord_list);

            //旋律
            for (auto& note : note_list) {
                if (note != lastNote) {
                    if (note != 0) {
                        midifile.addNoteOn(1, nowTm, 0, note, 120);
                    }
                    if (lastNote != 0) {
                        midifile.addNoteOff(1, nowTm, 0, lastNote, 120);
                    }
                    lastNote = note;
                }
            }

            //和弦
            script.nowChord = chord_list;
            //std::cout << "chord:" << chord_list.at(0) << d << std::endl;
            autochord::chordScript_setChord(script, script.nowChord);
            for (int i = 0; i < 16; ++i) {
                nowTm += 6;
                autochord::chordScript_resume(script);
                ++script.fragId;
            }
        }
    }
    if (lastNote != 0) {
        midifile.addNoteOff(1, nowTm, 0, lastNote, 90);
    }
    autochord::chordScript_stopAll(script);

    midifile.write(out);
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        return 0;
    }
    //getdir("../outputs/chordplay/", [](const char* pathname) {
    //    auto out = std::string(pathname) + ".chordplay.mid";
    //    buildMidi(out.c_str(), pathname);
    //});
    buildMidi(argv[2], argv[1]);
    return 0;
}
