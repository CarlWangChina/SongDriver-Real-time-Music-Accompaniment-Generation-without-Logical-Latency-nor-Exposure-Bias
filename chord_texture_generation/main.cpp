#include <emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include "chcpystring.h"
#include "chordScript.hpp"
#include "sectioner.hpp"

struct chordScript : autochord::chordScript_t, autochord::sectioner {
    int bpm, bps, fragId, baseTone;
    float averDelta;
    std::vector<int> nowChord;
} script;

inline void str2melody(const chcpy::string& str, std::vector<int>& melody) {
    auto notes_arr = str.replace(" ", "").replace("[", "").replace("]", "").split(",");
    for (auto it : notes_arr) {
        int note = it.toInt();
        melody.push_back(note);
    }
}

extern "C" {

EMSCRIPTEN_KEEPALIVE void chordScript_init(const char* path) {
    script.scriptPath = path;
    autochord::chordScript_init(script);
    autochord::sectioner_init(script, 16);
    script.fragId = 0;
    script.bps = 4;
    script.baseTone = 0;
    script.playNote_available = true;
    script.playNote = [&](int tone, int vel, int channel) {
        //std::cout << "playNote:" << tone << "," << vel << "," << channel << std::endl;
        EM_ASM({
            try {
                postMessage({"m" : "onPlayNote", "tone" : $0, "vel" : $1, "channel" : $2});
            } catch (e) {
                console.log(e);
            }
        },
               tone, vel, channel);
    };
    script.setIns = [&](int a, int b) {
        //std::cout << "setIns:" << a << "," << b << std::endl;
        EM_ASM({
            try {
                postMessage({"m" : "onSetIns", "channel" : $0, "instrumentId" : $1});
            } catch (e) {
                console.log(e);
            }
        },
               a, b);
    };
}
EMSCRIPTEN_KEEPALIVE void chordScript_setChord(const char* chord) {
    script.nowChord.clear();
    str2melody(chord, script.nowChord);
    //std::cout << "chord:" << chord << std::endl;
    autochord::chordScript_setChord(script, script.nowChord);
}
EMSCRIPTEN_KEEPALIVE void chordScript_step() {
    autochord::chordScript_resume(script);
    ++script.fragId;
}

int main() {
    EM_ASM({
        let chordScript_step = function() {
            Module._chordScript_step();
        };
        let chordScript_init = function(s) {
            var lengthBytes = lengthBytesUTF8(s) + 1;
            var stringOnWasmHeap = _malloc(lengthBytes);
            stringToUTF8(s, stringOnWasmHeap, lengthBytes);
            Module._chordScript_init(stringOnWasmHeap);
            _free(stringOnWasmHeap);
        };
        let chordScript_setChord = function(s) {
            var lengthBytes = lengthBytesUTF8(s) + 1;
            var stringOnWasmHeap = _malloc(lengthBytes);
            stringToUTF8(s, stringOnWasmHeap, lengthBytes);
            Module._chordScript_setChord(stringOnWasmHeap);
            _free(stringOnWasmHeap);
        };

        let status = false;

        function playLoop() {
            chordScript_step();  //下一步
            if (status) {
                setTimeout(playLoop, 500 / 16);
            }
        }

        function playSeq(str) {
            var raws = str.split("\n");
            var playIndex = 0;
            var playDatas = [];
            var lastNote = 0;
            var noteSubIndex = 0;
            for (var i = 0; i < raws.length; ++i) {
                var arr = raws[i].split("|");
                if (arr.length >= 2) {
                    var notes = arr[0].replace("[", "").replace("]", "").split(",");
                    playDatas.push([ notes, arr[1] ]);
                }
            }
            function playStep() {
                var beatId = Math.floor(playIndex / 16);
                var noteId = Math.floor(playIndex / 4) % 4;
                //console.log(beatId, noteId);
                if (beatId < playDatas.length) {
                    setTimeout(playStep, 500 / 16);
                    var beat = playDatas[beatId];
                    var nowMelody = beat[0][noteId];
                    if (nowMelody != lastNote) {
                        if (lastNote != 0) {
                            postMessage({"m" : "onPlayNote", "tone" : lastNote, "vel" : 0, "channel" : 0});
                        }
                        lastNote = nowMelody;
                        if (nowMelody != 0) {
                            postMessage({"m" : "onPlayNote", "tone" : nowMelody, "vel" : 90, "channel" : 0});
                        }
                    }
                    chordScript_setChord(beat[1]);
                    chordScript_step();  //下一步
                    ++playIndex;
                }
            }
            playStep();
        }

        let socket = new WebSocket("ws://127.0.0.1:8208/websocket");
        socket.onopen = function() {
            console.log("Connected!");
        };
        socket.onmessage = function(messageEvent) {
            playSeq(messageEvent.data);
        };

        onmessage = function(ev) {
            if (ev.data.m == "start") {
                status = true;
                playLoop();
            } else if (ev.data.m == "stop") {
                status = false;
            } else if (ev.data.m == "useScript") {
                chordScript_init(ev.data.src);
            } else if (ev.data.m == "toServer") {
                socket.send(ev.data.str);
            }
        };

        chordScript_init("scripts/sleepy.lua");
        postMessage({"m" : "ready"});
        console.log("chordscrpt ready");
    });
    return 0;
}
}
