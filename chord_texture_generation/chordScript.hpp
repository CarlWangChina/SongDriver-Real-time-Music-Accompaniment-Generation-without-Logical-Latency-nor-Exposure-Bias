//
// Created by admin on 2022/5/20.
//

#ifndef MIDILIB_CHORDSCRIPT_HPP
#define MIDILIB_CHORDSCRIPT_HPP

#include <math.h>
#include <array>
#include <set>
#include <string>
#include <vector>
#include "cyqueue_indexer.hpp"

extern "C" {
#include <lua/lauxlib.h>
#include <lua/lua.h>
#include <lua/lualib.h>
int luaopen_cjson(lua_State* l);
}

namespace autochord {
struct chordScript_t {
    std::string scriptPath;
    lua_State* lua_vm = nullptr;
    lua_State* lua_vm_coroutine = nullptr;

    std::function<void(int, int, int)> playNote;
    std::function<void(int, int)> setIns;

    std::vector<int> chord_notes;
    std::vector<int> chord_notes_real;

    std::set<std::pair<int, int> > playingNote;  //音符播放状态
    bool autoStopAll;                            //自动停止所有

    bool playNote_available;

    bool lua_vm_yield = false;

    int lastFragId;  //上次处理的id

    cyqueue_indexer_t chordHistoryQueue_indexer;
    std::array<int, 32> chordHistoryQueue;

    ~chordScript_t() {
        if (lua_vm != nullptr) {
            lua_close(lua_vm);
        }
    }
};

template <typename T>
void chordScript_init(T& self) {
    if (self.lua_vm != nullptr) {
        lua_close(self.lua_vm);
    }

    //初始化
    self.playingNote.clear();
    self.autoStopAll = false;
    self.lastFragId = -1;
    cyqueue_indexer_init(self.chordHistoryQueue_indexer, 32);
    for (int i = 0; i < 32; ++i) {
        self.chordHistoryQueue[i] = 0;
    }

    printf("chordGen:chordScript_lua_vm_init\n");
    self.lua_vm_yield = false;
    self.playNote_available = false;
    printf("chordGen:chordScript_lua_vm_init:create state\n");
    self.lua_vm = luaL_newstate();
    printf("chordGen:chordScript_lua_vm_init:luaL_openlibs\n");
    luaL_openlibs(self.lua_vm);
    //加入lua-cjson
    luaopen_cjson(self.lua_vm);
    lua_setglobal(self.lua_vm, "cjson");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        const char* str = luaL_checkstring(L, 1);
        printf("chordGen_script_log:%s\n", str);
        return 0;
    });
    lua_setglobal(self.lua_vm, "log_print");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        const char* str = luaL_checkstring(L, 1);
        printf("chordGen_script_log:%s\n", str);
        return 0;
    });
    lua_setglobal(self.lua_vm, "log_error");

    printf("chordGen:chordScript_lua_vm_init:load file:%s\n",
           self.scriptPath.c_str());
    //初始化执行
    luaL_loadfile(self.lua_vm, self.scriptPath.c_str());
    if (lua_pcall(self.lua_vm, 0, LUA_MULTRET, 0)) {
        auto res = lua_tostring(self.lua_vm, -1);
        printf("chordGen:script_error:%s\n", res);
        lua_pop(self.lua_vm, 1);
    }

    printf("chordGen:chordScript_lua_vm_init:load functions\n");
    //载入后续函数
    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        return lua_yield(L, 1);
    });
    lua_setglobal(self.lua_vm, "sleepSec");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        //int tone, int vel, int channel
        int tone = luaL_checkinteger(L, 2);
        int vel = luaL_checkinteger(L, 3);
        int channel = luaL_checkinteger(L, 4);
        if (self->playNote_available) {
            if (vel > 0) {
                if (self->playingNote.find(std::pair<int, int>(tone, channel)) != self->playingNote.end()) {
                    self->playNote(tone, 0, channel);
                }
            }
            self->playNote(tone, vel, channel);
        }
        //保存音符状态
        if (vel > 0) {
            self->playingNote.insert(std::pair<int, int>(tone, channel));
        } else {
            self->playingNote.erase(std::pair<int, int>(tone, channel));
        }
        lua_pushboolean(L, true);
        return 1;
    });
    lua_setglobal(self.lua_vm, "play");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        lua_pushinteger(L, self->chord_notes_real.size());
        return 1;
    });
    lua_setglobal(self.lua_vm, "playListSize");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        int t = luaL_checkinteger(L, 2);
        if (!self->chord_notes_real.empty()) {
            auto tone = self->chord_notes_real.at(0) % 12;
            while (tone < t) {
                tone += 12;
            }
            auto delta = tone - self->chord_notes_real.at(0);
            lua_pushinteger(L, delta);
            return 1;
        }
        return 0;
    });
    lua_setglobal(self.lua_vm, "shiftPlayList");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        //int tone, int vel, int channel
        try {
            int toneId = luaL_checkinteger(L, 2);
            int tone = self->chord_notes_real.at(toneId - 1);
            int otone = tone;
            int vel = luaL_checkinteger(L, 3);
            int channel = luaL_checkinteger(L, 4);
            if (lua_isinteger(L, 5)) {
                tone += lua_tointeger(L, 5);
            }
            if (self->playNote_available) {
                if (vel > 0) {
                    if (self->playingNote.find(std::pair<int, int>(tone, channel)) != self->playingNote.end()) {
                        self->playNote(tone, 0, channel);
                    }
                }
                self->playNote(tone, vel, channel);
            }
            if (self->playNote_available) {
                self->playNote(tone, vel, channel);
            }
            //保存音符状态
            //std::cout << "tone:" << tone << " " << self->chord_notes_real.at(0) << std::endl;
            if (vel > 0) {
                self->playingNote.insert(std::pair<int, int>(tone, channel));
            } else {
                self->playingNote.erase(std::pair<int, int>(tone, channel));
            }
            lua_pushboolean(L, true);
        } catch (...) {
            lua_pushboolean(L, false);
        }
        return 1;
    });
    lua_setglobal(self.lua_vm, "playIndex");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        chordScript_stopAll(*self);
        lua_pushboolean(L, true);
        return 1;
    });
    lua_setglobal(self.lua_vm, "stopAll");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        if (!lua_isboolean(L, 2))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        self->autoStopAll = lua_toboolean(L, 2);
        lua_pushboolean(L, true);
        return 1;
    });
    lua_setglobal(self.lua_vm, "setAutoStopAll");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        int a = luaL_checkinteger(L, 2);
        int b = luaL_checkinteger(L, 3);
        if (self->playNote_available) {
            self->setIns(a, b);
        }
        lua_pushboolean(L, true);
        return 1;
    });
    lua_setglobal(self.lua_vm, "setIns");

    lua_pushcfunction(self.lua_vm, [](lua_State* L) -> int {
        if (!lua_isuserdata(L, 1))
            return 0;
        auto self = (T*)lua_touserdata(L, 1);
        int pos = luaL_checkinteger(L, 2);
        int index = cyqueue_indexer_last(self->chordHistoryQueue_indexer, pos);
        try {
            int data = self->chordHistoryQueue.at(index);
            lua_pushinteger(L, data);
            return 1;
        } catch (...) {
            return 0;
        }
    });
    lua_setglobal(self.lua_vm, "getChordHistory");

    self.lua_vm_coroutine = lua_newthread(self.lua_vm);  //创建协程
    lua_getglobal(self.lua_vm_coroutine, "main");        // 协程函数入栈

    printf("chordGen:chordScript_lua_vm_init:call main()\n");
    if (lua_isfunction(self.lua_vm_coroutine, 1)) {
        lua_pushlightuserdata(self.lua_vm_coroutine, &self);  //参数
        int nres = 0;
        auto re = lua_resume(self.lua_vm_coroutine, NULL, 1, &nres);  //启动
        self.lua_vm_yield = (re == LUA_YIELD);                        //协程暂停了
        if (!self.lua_vm_yield) {
            //获取错误
            const char* res;
            if (re == LUA_ERRRUN) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:init LUA_ERRRUN:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRSYNTAX) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:init LUA_ERRSYNTAX:%s\n",
                       res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRMEM) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:init LUA_ERRMEM:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRERR) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:init LUA_ERRERR:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            }
        } else {
            printf("chordGen:chordScript_lua_vm_init coroutine yield\n");
            printf("chordGen:chordScript_lua_vm_init processing start\n");
        }
        if (nres > 0) {
            lua_pop(self.lua_vm_coroutine, nres);
        }
    }
    printf("chordGen:chordScript_lua_vm_init success\n");
}

template <typename T>
void chordScript_stopAll(T& self) {
    if (self.playNote_available) {
        for (auto& it : self.playingNote) {
            int tone = it.first;
            int channel = it.second;
            self.playNote(tone, 0, channel);
        }
    }
    self.playingNote.clear();
}

template <typename T>
void chordScript_setChord(T& self, const std::vector<int>& notes) {
    self.chord_notes = notes;
    self.chord_notes_real.clear();
    int last = 0;
    for (auto it : notes) {
        int n = it + self.baseTone;
        while (n < last) {
            n += 12;
        }
        last = n;
        self.chord_notes_real.push_back(n);
    }
    if (!notes.empty()) {
        int note = notes.at(0);
        if (note > 0) {
            int lastNote = self.chordHistoryQueue.at(self.chordHistoryQueue_indexer.index);
            if (lastNote != note) {
                cyqueue_indexer_next(self.chordHistoryQueue_indexer);
                self.chordHistoryQueue.at(self.chordHistoryQueue_indexer.index) = note;
            }
        }
    }
}

template <typename T>
void chordScript_resume(T& self) {
    if (self.lua_vm && self.lua_vm_coroutine && self.lua_vm_yield) {
        //释放资源
        //每小节自动停止
        auto secId_now = ((int)floor(self.fragId / (16. * self.bps)));
        auto secId_last = ((int)floor(self.lastFragId / (16. * self.bps)));
        if (self.autoStopAll) {
            if (secId_now != secId_last) {
                chordScript_stopAll(self);
            }
        }
        self.lastFragId = self.fragId;

        //创建表
        lua_createtable(self.lua_vm_coroutine, 0, 6);  //6个map元素

        //id
        lua_pushinteger(self.lua_vm_coroutine, self.fragId);
        lua_setfield(self.lua_vm_coroutine, -2, "fragId");

        //调性
        lua_pushinteger(self.lua_vm_coroutine, self.baseTone);
        lua_setfield(self.lua_vm_coroutine, -2, "baseTone");

        //和弦
        int len = self.chord_notes.size();
        lua_createtable(self.lua_vm_coroutine, len, 0);  //len个数组元素
        int key = 1;
        for (auto it : self.chord_notes) {
            lua_pushinteger(self.lua_vm_coroutine, it);
            lua_rawseti(self.lua_vm_coroutine, -2, key);
            ++key;
        }
        lua_setfield(self.lua_vm_coroutine, -2, "chord_notes");

        len = self.chord_notes_real.size();
        lua_createtable(self.lua_vm_coroutine, len, 0);  //len个数组元素
        key = 1;
        for (auto it : self.chord_notes_real) {
            lua_pushinteger(self.lua_vm_coroutine, it);
            lua_rawseti(self.lua_vm_coroutine, -2, key);
            ++key;
        }
        lua_setfield(self.lua_vm_coroutine, -2, "chord_notes_real");

        //变化率
        lua_pushnumber(self.lua_vm_coroutine, self.averDelta);
        lua_setfield(self.lua_vm_coroutine, -2, "averDelta");

        lua_pushinteger(self.lua_vm_coroutine, self.bpm);
        lua_setfield(self.lua_vm_coroutine, -2, "bpm");

        lua_pushinteger(self.lua_vm_coroutine, self.bps);
        lua_setfield(self.lua_vm_coroutine, -2, "bps");

        //启动
        int nres = 0;
        auto re = lua_resume(self.lua_vm_coroutine, NULL, 1, &nres);
        self.lua_vm_yield = (re == LUA_YIELD);
        if (!self.lua_vm_yield) {
            //获取错误
            const char* res;
            if (re == LUA_ERRRUN) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:LUA_ERRRUN:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRSYNTAX) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:LUA_ERRSYNTAX:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRMEM) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:LUA_ERRMEM:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            } else if (re == LUA_ERRERR) {
                res = lua_tostring(self.lua_vm_coroutine, -1);
                printf("chordGen:LUA_ERRERR:%s\n", res);
                lua_pop(self.lua_vm_coroutine, 1);
            }
        }
        if (nres > 0) {
            lua_pop(self.lua_vm_coroutine, nres);
        }
    }
}
}  // namespace autochord

#endif  //MIDILIB_CHORDSCRIPT_HPP
