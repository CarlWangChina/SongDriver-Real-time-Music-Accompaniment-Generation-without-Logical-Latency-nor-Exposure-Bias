#pragma once
#include "hmm.h"
#include "melody2chord.h"
#include "seq2id.h"
namespace chcpy::predict {

template <melody2chord::chord_map_c chord_map_t,
          seq2id::dict_c melody_dict_t,
          chord2id::dict_c chord_dict_t,
          typename model_t>
inline void genChord(chord_map_t& chord_map,      //和弦匹配表
                     melody_dict_t& melody_dict,  //旋律字典
                     chord_dict_t& chord_dict,    //和弦字典
                     model_t& model,
                     const melody2chord::musicSection_t& buffer,
                     midiSearch::chord_t& chord,
                     bool fixEmpty = true) {
    seq2id::melody_t buffer_melody, melody_seq;
    std::vector<int> chord_id;
    int minNote = 9999999;
    for (auto note : buffer.melody) {  //转换为14进制级数
        if (note > 0 && note < minNote) {
            minNote = note;
        }
        int noteLevel = chcpy::melody2chord::getToneLevelDelta(note, buffer.chord_base, 0);
        buffer_melody.push_back(noteLevel);
    }
    if (minNote == 9999999) {
        minNote = 24;  //默认值
    }
    seq2id::melody2seq(melody_dict, buffer_melody, melody_seq);  //每四个一组
    hmm::predict(model, melody_seq, chord_id);                   //hmm预测
#ifdef CHCPY_DEBUG
    printf("%d_%s(w=%f) len(seq)=%d\n",
           buffer.chord_base,
           buffer.chord_name.c_str(),
           buffer.weight,
           melody_seq.size());
    printf("旋律\t\t模型输出\t实际结果\n");
    auto melody_it = buffer.melody.begin();
#endif
    const int idnull = chord2id::get(chord_dict, "null");
    bool fail = true;
    for (auto id : chord_id) {
        if (id != idnull) {
            fail = false;
            break;
        }
    }
    for (auto id : chord_id) {
#ifdef CHCPY_DEBUG
        int melody_it_count = 0;
        printf("[");
        for (; (melody_it != buffer.melody.end() && ++melody_it_count <= 4); ++melody_it) {
            printf("%d ", *melody_it);
        }
        printf("]\t");
#endif
        //转换为和弦
        auto chord14 = chord2id::get(chord_dict, id);  //搜索
        std::vector<int> singleChord;
#ifdef CHCPY_DEBUG
        printf("[");
#endif
        if (id != idnull || (fail && fixEmpty)) {
            chcpy::stringlist note14s;
            if (fail) {  //hmm失效，回退到申克分析
                note14s = {"0", "4", "8"};
            } else {
                note14s = chcpy::string(chord14.c_str()).split("-");  //分割
            }
            int B = buffer.chord_base;  //片段根音
            int last = -1;
            int maxNote = -1;
            for (auto note14_str : note14s) {  //遍历
#ifdef CHCPY_DEBUG
                printf("%s ", note14_str.c_str());
#endif
                int A = note14_str.toInt();                               //A的原始值（14进制）
                int tone = melody2chord::getToneFromLevelDelta(A, B, 0);  //实际音阶（12平均律）
                while (tone < last) {
                    tone += 12;
                }
                last = tone;
                if (maxNote < tone) {
                    maxNote = tone;
                }
                singleChord.push_back(tone);
            }
            int toneShift = (1 + (maxNote - minNote) / 12) * 12;
            for (auto& note : singleChord) {
                note -= toneShift;
            }
        }
#ifdef CHCPY_DEBUG
        printf("]\t(");
        for (auto note : singleChord) {
            printf("%d ", note);
        }
        printf(")\n");
#endif
        chord.push_back(singleChord);
    }
}

//流程：
//切分旋律为和弦
//将切分结果转换为14进制级数
//将14进制级数每四个一组切分，并转换为hmm所需id
//将id送入hmm模型预测
//将预测结果由14进制级数转换为12平均律
template <melody2chord::chord_map_c chord_map_t,
          seq2id::dict_c melody_dict_t,
          chord2id::dict_c chord_dict_t,
          typename model_t>
inline void gen(
    chord_map_t& chord_map,          //和弦匹配表
    melody_dict_t& melody_dict,      //旋律字典
    chord_dict_t& chord_dict,        //和弦字典
    model_t& model,                  //hmm模型
    const seq2id::melody_t& melody,  //输入旋律
    midiSearch::chord_t& chord       //返回和弦
) {
    std::list<melody2chord::musicSection_t> segs = chcpy::melody2chord::getMusicSection(  //切分
        chord_map,
        melody,
        4, 0.5, 1.0);
    chord.clear();
    for (auto seg : segs) {
        genChord(chord_map, melody_dict, chord_dict, model, seg, chord);
    }
}

}  // namespace chcpy::predict