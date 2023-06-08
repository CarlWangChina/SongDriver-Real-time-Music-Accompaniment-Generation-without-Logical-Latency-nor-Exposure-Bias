#pragma once
#include "bayes.h"
#include "chordNext.h"
#include "hmm_gpu.h"
#include "predict.h"
namespace chcpy::rtmtc {

struct rtmtc_t {
    melody2chord::musicSection_t buffer{};
    midiSearch::chord_t chord{};
    midiSearch::melody_t notes{};
    midiSearch::melody_t lastChord{};
    midiSearch::melody_t historyEndChord{};
    float act = 0.5;
    float sectionWeight = 1.0;
    int times = 1;
    int maxNoteNum = 64;
    int octave = 0;
    int historyOctave = 0;
    bool updated = false;
    bool fixEmpty = false;
};
template <typename T>
concept rtmtc_c = requires(T a) {
    a.act = 0;
    a.sectionWeight = 0;
    a.times = 0;
    a.updated = false;
    a.buffer;
    a.chord;
    a.notes;
    a.octave;
    a.historyOctave;
    a.lastChord;
    a.historyEndChord;
    a.fixEmpty;
};

inline int getEndChord(const midiSearch::chord_t& chord, midiSearch::melody_t& outChord) {
    int res = 0;
    outChord.clear();
    for (auto it = chord.rbegin(); it != chord.rend(); ++it) {
        if (it->size() > 0) {
            res = it->at(0) / 12;
            outChord = *it;
            break;
        }
    }
    return res;
}

template <melody2chord::chord_map_c chord_map_t,
          rtmtc_c rtmtc_type,
          typename hmm_type,
          seq2id::dict_c dict_seq_t,
          chord2id::dict_c dict_chord_t>
inline bool pushSection(rtmtc_type& self,
                        const chord_map_t& chord_map,
                        hmm_type& model,
                        dict_seq_t& dict_melody,
                        dict_chord_t& dict_chord,
                        const melody2chord::musicSection_t& section) {
    self.updated = false;
    if (self.buffer.melody.empty()) {
        //第一次使用
        self.buffer = section;
    } else {
        //检测合并
        auto meg = melody2chord::merge(chord_map, self.buffer, section, self.act, self.times);
        float nowWeight = self.buffer.weight + section.weight + self.sectionWeight;
        if (nowWeight >= meg.weight && meg.melody.size() < self.maxNoteNum) {
            //合并
            self.buffer = std::move(meg);
        } else {
            if (!self.buffer.melody.empty()) {
                //返回
                self.chord.clear();
                predict::genChord(chord_map,
                                  dict_melody,
                                  dict_chord,
                                  model,
                                  self.buffer,
                                  self.chord,
                                  self.fixEmpty);
                midiSearch::melody_t lastChord;
                int octave = getEndChord(self.chord, lastChord);
                if (octave > 0) {
                    self.historyOctave = octave;
                }
                if (!lastChord.empty()) {
                    self.historyEndChord = lastChord;
                }
                self.updated = true;
            }
            self.buffer = section;
        }
    }
    return self.updated;
}

template <melody2chord::chord_map_c chord_map_t,
          rtmtc_c rtmtc_type,
          typename hmm_type,
          seq2id::dict_c dict_seq_t,
          chord2id::dict_c dict_chord_t>
inline bool pushNote(rtmtc_type& self,
                     const chord_map_t& chord_map,
                     hmm_type& model,
                     dict_seq_t& dict_melody,
                     dict_chord_t& dict_chord,
                     int note) {
    self.notes.push_back(note);
    if (self.notes.size() >= 4) {
        pushSection(
            self,
            chord_map,
            model,
            dict_melody,
            dict_chord,
            melody2chord::buildMusicSection(
                chord_map,
                self.notes,
                self.act,
                self.times));
        self.notes.clear();
        return true;
    }
    return false;
}

template <melody2chord::chord_map_c chord_map_t,
          rtmtc_c rtmtc_type,
          typename hmm_type,
          seq2id::dict_c dict_seq_t,
          chord2id::dict_c dict_chord_t>
inline void buildRealtimeBuffer(const chord_map_t& chord_map,
                                hmm_type& model,
                                dict_seq_t& dict_melody,
                                dict_chord_t& dict_chord,
                                activeBuffer& buf,
                                rtmtc_type& now,
                                midiSearch::chord_t& newChord,
                                std::vector<activeBuffer::chordtime>& res) {
    std::vector<std::string> newChord_str;
    predict::genChord(chord_map,
                      dict_melody,
                      dict_chord,
                      model,
                      now.buffer,
                      newChord,
                      now.fixEmpty);

    midiSearch::melody_t lastChord;
    now.octave = now.historyOctave;
    int octave = getEndChord(newChord, lastChord);
    if (octave > 0) {
        now.octave = octave;
    }
    if (!lastChord.empty()) {
        now.lastChord = lastChord;
    }

    //转换为字符串
    for (auto& chord : newChord) {
        std::vector<std::string> buffer;
        for (auto& it : chord) {
            buffer.push_back(it == 0 ? "0" : chcpy::string::number(it % 12));
        }
        auto str = join(buffer, "-");
        //printf("%s\n", str.c_str());
        newChord_str.push_back(str);
    }
    buf.buildRealtimeBuffer(newChord_str, res);
}

inline midiSearch::melody_t str2chord(const chcpy::string& chord, int oct) {
    midiSearch::melody_t outChord;
    int lastNote = -1;
    auto arr = chord.split("-");
    for (auto& it : arr) {
        if (!it.empty()) {
            int note = it.toInt() + oct * 12;
            if (lastNote != -1) {
                while (note < lastNote) {
                    note += 12;
                }
            }
            lastNote = note;
            outChord.push_back(note);
        }
    }
    return outChord;
}

template <int bayeslen = 5, chcpy::chordNext::dict_c dict_type>
inline midiSearch::melody_t genChord(dict_type& dict,
                                     bayes::bayes_predict_t<bayeslen>& model,
                                     const midiSearch::melody_t& lastChord,
                                     int oct,
                                     const std::vector<activeBuffer::chordtime>& res) {
    auto max_id = chcpy::chordNext::predictNext(dict, model, res);
    auto it = dict.id_chord.find(max_id);
    if (it != dict.id_chord.end()) {
        string chord = std::get<0>(it->second);
        //转换为数字
        return str2chord(chord, oct);
    } else {
        return lastChord;
    }
}

}  // namespace chcpy::rtmtc