#pragma once
#include <list>
#include <map>
#include <string>
#include <vector>
#include "chcpystring.h"
#include "search.h"

namespace chcpy {

struct activeBuffer {
    using chordtime = std::tuple<std::string, int>;
    std::string nowChord{};        //当前状态
    int nowChord_time = 0;         //当前状态持续时间
    std::list<chordtime> history;  //历史记录
    int history_max = 5;
    inline void pushChord(const std::string& chord) {
        if (chord.empty()) {
            return;
        }
        if (chord == nowChord) {
            nowChord_time += 1;
        } else {
            if (!nowChord.empty()) {
                history.push_back(std::make_tuple(nowChord, nowChord_time));
                if (history.size() > history_max) {
                    history.pop_front();
                }
            }
            nowChord = chord;
            nowChord_time = 1;
        }
    }
    inline void pushChord(const std::vector<int>& chord) {
        std::vector<std::string> buffer;
        for (auto& it : chord) {
            buffer.push_back(it == 0 ? "0" : chcpy::string::number(it % 12));
        }
        auto str = join(buffer, "-");
        pushChord(str);
    }
    inline void pushChord(const midiSearch::chord_t& chord) {
        for (auto& it : chord) {
            pushChord(it);
        }
    }
    inline void buildRealtimeBuffer(const std::vector<std::string>& realtime,
                                    std::vector<chordtime>& res) const {
#define checkSize()                  \
    if (res.size() >= history_max) { \
        goto end;                    \
    }
        res.clear();
        std::string last;
        int last_count = 0;
        for (auto it = realtime.rbegin(); it != realtime.rend(); ++it) {
            auto& str = *it;
            if (!str.empty()) {
                if (str != last) {
                    if (!last.empty()) {
                        res.push_back(std::make_tuple(last, last_count));
                        checkSize();
                    }
                    last_count = 0;
                }
                ++last_count;
                last = str;
            }
        }
        if (last_count != 0 && !last.empty()) {
            if (!nowChord.empty() && nowChord_time != 0) {
                if (nowChord == last) {
                    res.push_back(std::make_tuple(last, last_count + nowChord_time));
                    checkSize();
                } else {
                    res.push_back(std::make_tuple(last, last_count));
                    checkSize();
                    res.push_back(std::make_tuple(nowChord, nowChord_time));
                    checkSize();
                }
            } else {
                res.push_back(std::make_tuple(last, last_count));
                checkSize();
            }
        } else {
            if (!nowChord.empty() && nowChord_time != 0) {
                res.push_back(std::make_tuple(nowChord, nowChord_time));
                checkSize();
            }
        }
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            res.push_back(std::make_tuple(std::get<0>(*it), std::get<1>(*it)));
            checkSize();
        }
    end:
        //for (auto it : res) {
        //    printf("ab:%s = %d\n", std::get<0>(it).c_str(), std::get<1>(it));
        //}
        while (res.size() > history_max) {
            res.pop_back();
        }
#undef checkSize
    }
};

template <typename T>
concept activeBuffer_c = requires(T a) {
    a.buildRealtimeBuffer(std::vector<std::string>(),
                          std::vector<activeBuffer::chordtime>());
    a.pushChord(std::string());
};

}  // namespace chcpy