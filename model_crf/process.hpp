#pragma once
#include "crf.hpp"
namespace autochord {

struct musicBuffer {
    std::list<std::string> melody{};
    std::list<std::string> chord{};
    std::list<std::string> chord_section{};
    int id = 0;
    musicBuffer() {
        for (int i = 0; i < 8; ++i) {
            melody.push_back("");
        }
        for (int i = 0; i < 4; ++i) {
            chord.push_back("");
        }
        for (int i = 0; i < 16; ++i) {
            chord_section.push_back("");
        }
    }
    void pushNote(int n) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", n);
        melody.push_back(buf);
        melody.pop_front();
    }
    void pushMelody(int n_1, int n_2, int n_3, int n_4) {
        pushNote(n_1);
        pushNote(n_2);
        pushNote(n_3);
        pushNote(n_4);
        ++id;
    }
    void pushChord(const std::string& ch) {
        chord.push_back(ch);
        chord.pop_front();
        chord_section.push_back(ch);
        chord_section.pop_front();
    }
};

}  // namespace autochord