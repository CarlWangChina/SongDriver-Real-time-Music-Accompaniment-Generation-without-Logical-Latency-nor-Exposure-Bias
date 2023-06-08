#pragma once
#include <map>
#include <unordered_map>
#include "activeBuffer.h"
#include "bayes.h"
#include "chcpystring.h"
namespace chcpy::chordNext {

struct dict_t {
    std::unordered_map<std::string, std::map<int, int>> chord_id;
    std::map<int, std::tuple<std::string, int>> id_chord;
    inline int getIndex(const std::string& chord, int tm) const {
        auto it = chord_id.find(chord);
        if (it != chord_id.end()) {
            auto tm_it = it->second.find(tm);
            if (tm_it != it->second.end()) {
                return tm_it->second;
            } else {
                if (it->second.empty()) {
                    return -1;
                }
                auto low = it->second.lower_bound(tm);
                if (low == it->second.end()) {
                    return it->second.begin()->second;
                } else {
                    auto up = low;
                    up++;
                    if (up == it->second.end()) {
                        return low->second;
                    } else {
                        if (std::abs(up->first - tm) < std::abs(low->first - tm)) {
                            return up->second;
                        } else {
                            return low->second;
                        }
                    }
                }
            }
        }
        return -1;
    }
    inline int getIndex(const std::string& chordtime) {
        chcpy::string str = chordtime;
        auto chord_time = str.split("=");
        if (chord_time.size() >= 2) {
            return getIndex(chord_time[0].trimmed(), chord_time[1].trimmed().toInt());
        } else {
            return -1;
        }
    }
    inline void load(const char* path) {
        auto fp = fopen(path, "r");
        if (fp) {
            chcpy::string line;
            char buf[128];
            while (!feof(fp)) {
                bzero(buf, sizeof(buf));
                fgets(buf, sizeof(buf), fp);
                line = buf;
                line = line.trimmed();
                auto data_id = line.split("|");
                if (data_id.size() >= 2) {
                    auto chord_time = data_id[0].split("=");
                    if (chord_time.size() >= 2) {
                        int id = data_id[1].trimmed().toInt();
                        auto chord_str = chord_time[0].trimmed();
                        auto chord_tm = chord_time[1].trimmed().toInt();
                        chord_id[chord_str][chord_tm] = id;
                        id_chord[id] = std::tuple<std::string, int>(chord_str, chord_tm);
                    }
                }
            }
            fclose(fp);
        }
    }
};

template <typename T>
concept dict_c = requires(T a) {
    a.getIndex(std::string(), int());
};

template <int bayeslen = 5, dict_c dict_type>
inline int predictNext(dict_type& dict,
                       bayes::bayes_predict_t<bayeslen>& model,
                       const std::vector<activeBuffer::chordtime>& res) {
    std::array<int, bayeslen> arr;
    for (int i = 0; i < bayeslen; ++i) {
        arr[i] = -1;
    }
#ifdef CHCPY_DEBUG
    printf("输入数据：\n");
#endif
    int i = 0;
    for (auto& it : res) {
#ifdef CHCPY_DEBUG
        printf("%s=%d\n", std::get<0>(it).c_str(), std::get<1>(it));
#endif
        auto index = dict.getIndex(std::get<0>(it), std::get<1>(it));
        arr[i++] = index;
        if (i >= bayeslen) {
            break;
        }
    }
#ifdef CHCPY_DEBUG
    printf("输入序列：\n");
    for (auto it : arr) {
        printf("%d ", it);
    }
    printf("\n");
#endif
    auto prob = model.getProb(arr);
    float max_value = 0;
    int max_id = -1;
#ifdef CHCPY_PROB_DEBUG
    printf("概率：\n");
#endif
    for (auto& it : prob) {
#ifdef CHCPY_PROB_DEBUG
        printf("%d->%f\n", it.first, it.second);
#endif
        if (it.second > max_value) {
            max_value = it.second;
            max_id = it.first;
        }
    }
    return max_id;
}

template <int bayeslen = 5, dict_c dict_type>
inline void trainLine(dict_type& dict,
                      bayes::bayes_train_t<bayeslen>& model,
                      const chcpy::string& input) {
    std::vector<int> ids;
    std::array<int, bayeslen> arr;
    auto chords = input.trimmed().split(" ");
    for (auto& it : chords) {
        auto str = it.trimmed();
        if (!str.empty()) {
            ids.push_back(dict.getIndex(str));
        }
    }
    int len = ids.size();
    for (int i = 0; i < len; ++i) {
        int val = ids[i];
        for (int j = 1; j <= bayeslen; ++j) {  //前bayeslen个（含第bayeslen个）
            int index = i - j;
            if (index < 0) {
                arr[j - 1] = -1;
            } else {
                arr[j - 1] = ids[index];
            }
        }
        model.add(arr, val);
    }
}

}  // namespace chcpy::chordNext