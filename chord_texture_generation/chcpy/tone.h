#pragma once
#include <math.h>
#include <memory.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include "cJSON.h"
#include "search.h"

namespace chcpy::tone {

template <typename T, int n>  //pearson相关系数
inline double pearson(const std::array<T, n>& inst1, const std::array<T, n>& inst2) {
    double pearson = n * std::inner_product(inst1.begin(), inst1.end(), inst2.begin(), 0.0) -
                     std::accumulate(inst1.begin(), inst1.end(), 0.0) *
                         std::accumulate(inst2.begin(), inst2.end(), 0.0);

    double temp1 = n * std::inner_product(inst1.begin(), inst1.end(), inst1.begin(), 0.0) -
                   pow(std::accumulate(inst1.begin(), inst1.end(), 0.0), 2.0);

    double temp2 = n * std::inner_product(inst2.begin(), inst2.end(), inst2.begin(), 0.0) -
                   pow(std::accumulate(inst2.begin(), inst2.end(), 0.0), 2.0);

    temp1 = sqrt(temp1);
    temp2 = sqrt(temp2);
    pearson = pearson / (temp1 * temp2);

    return pearson;
}

struct toneConfig_t {
    std::string name{};
    int baseTone = 0;
    std::array<float, 12> weights;
    inline double getPearson(const std::array<float, 12>& arr) const {
        return pearson<float, 12>(arr, weights);
    }
    inline bool set(const char* str) {
        bool haveBaseTone = false;
        bool haveWeights = false;
        bool haveName = false;
        auto obj = cJSON_Parse(str);
        if (obj) {
            if (obj->type == cJSON_Object) {  //如果是对象
                auto line = obj->child;       //开始遍历
                while (line) {
                    if (strcmp(line->string, "baseTone") == 0) {  //名称为baseTone
                        if (line->type == cJSON_Number) {         //类型为数字
                            baseTone = line->valueint;            //设置baseTone
                            haveBaseTone = true;
                        }
                    } else if (strcmp(line->string, "weights") == 0) {  //获取weights
                        if (line->type == cJSON_Array) {                //类型为数组
                            int index = 0;
                            auto prob = line->child;  //开始遍历
                            while (prob) {
                                if (prob->type == cJSON_Number) {  //类型为数字
                                    weights[index] = prob->valueint;
                                    haveWeights = true;
                                    ++index;
                                    if (index >= 12) {
                                        break;
                                    }
                                }
                                prob = prob->next;
                            }
                        }
                    } else if (strcmp(line->string, "name") == 0) {  //获取name
                        if (line->type == cJSON_String) {            //类型为str
                            name = line->valuestring;
                            haveName = true;
                        }
                    }
                    line = line->next;
                }
                cJSON_Delete(obj);
            }
        }
        return haveBaseTone && haveWeights && haveName;
    }
};

using tonemap_t = std::vector<toneConfig_t>;

inline auto createToneConfig(const char* path) {
    tonemap_t res;
    auto fp = fopen(path, "r");
    toneConfig_t item;

    char buf[1024];
    if (fp) {
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            if (item.set(buf)) {
                res.push_back(item);
            }
        }
        fclose(fp);
    }
    return res;
}

inline auto getToneFromLen(
    const tonemap_t& self,
    const std::array<float, 12>& num,
    double* problist = nullptr) {
    size_t len = self.size();
    double res[len];
    if (problist == nullptr) {
        problist = &res[0];
    }
#pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
        problist[i] = std::abs(self[i].getPearson(num));
    }
    int max_id = -1;
    double max_val = -1;
    for (size_t i = 0; i < len; ++i) {
        double val = problist[i];
        if (val > max_val) {
            max_val = val;
            max_id = i;
        }
    }
    return std::make_tuple(max_id, max_val);
}

struct melodyChunk_t {
    midiSearch::mu_melody_t melody{};
    std::array<float, 12> count{};
    double weight = 0;
    double aweight = 0;  //权重乘长度
    const toneConfig_t* tone = nullptr;
};

inline void initMelodyChunk(const tonemap_t& self, melodyChunk_t& res) {
    for (auto& it : res.count) {
        it = 0;
    }
    int gcount = 0;
    for (auto& notes : res.melody) {
        for (auto& note : notes) {
            if (note != 0) {
                res.count[note % 12] += 1;
                gcount += 1;
            }
        }
    }
    if (gcount == 0) {  //没有音符
        res.tone = nullptr;
        res.weight = 999999;
        res.aweight = res.weight * res.melody.size();
    } else {
        auto w = chcpy::tone::getToneFromLen(self, res.count);
        res.tone = &self[std::get<0>(w)];
        res.weight = std::get<1>(w);
        res.aweight = res.weight * res.melody.size();
    }
}

inline auto buildMelodyChunk(const tonemap_t& self, const midiSearch::mu_melody_t& melody) {
    melodyChunk_t res;
    res.melody = melody;
    initMelodyChunk(self, res);
    return res;
}

inline auto mergeMelodyChunk(const tonemap_t& self,
                             const melodyChunk_t& A,
                             const melodyChunk_t& B) {
    melodyChunk_t res;
    res.melody.clear();
    for (auto it : A.melody) {
        res.melody.push_back(it);
    }
    for (auto it : B.melody) {
        res.melody.push_back(it);
    }
    initMelodyChunk(self, res);
    return res;
}

inline void getToneSection(const tonemap_t& tone_map,
                           std::list<melodyChunk_t>& toneChunks,
                           int minSectionLen) {
    while (1) {
        int processNum = 0;
        auto it = toneChunks.begin();
        while (it != toneChunks.end()) {  //贪心合并
            auto next_it = it;
            ++next_it;
            if (next_it != toneChunks.end()) {
                //存在下一个，检测合并
                auto& now = *it;
                auto& next = *next_it;
                auto merged = mergeMelodyChunk(tone_map, now, next);
                float nowWeight = now.aweight + next.aweight;
                if (now.tone == next.tone || nowWeight <= merged.aweight) {
                    //确认合并
                    now = std::move(merged);
                    toneChunks.erase(next_it);
                    ++processNum;
                } else {
                    ++it;
                }
            } else {
                break;
            }
        }
        it = toneChunks.begin();
        while (it != toneChunks.end()) {  //强制合并
            if (it->melody.size() < minSectionLen) {
                auto next_it = it;
                ++next_it;
                auto& now = *it;
                auto& next = *next_it;
                auto pre_it = it;
                if (it == toneChunks.begin()) {
                    //开头
                    auto merged = mergeMelodyChunk(tone_map, now, next);
                    now = std::move(merged);
                    toneChunks.erase(next_it);
                    ++processNum;
                } else {
                    --pre_it;
                    auto& pre = *pre_it;
                    if (next_it == toneChunks.end()) {
                        //结尾
                        auto merged = mergeMelodyChunk(tone_map, pre, now);
                        now = std::move(merged);
                        toneChunks.erase(pre_it);
                        ++processNum;
                    } else {
                        //中间
                        auto merged_next = mergeMelodyChunk(tone_map, now, next);
                        auto merged_pre = mergeMelodyChunk(tone_map, pre, now);
                        if (merged_next.weight > merged_pre.weight) {
                            now = std::move(merged_next);
                            toneChunks.erase(next_it);
                            ++processNum;
                        } else {
                            now = std::move(merged_pre);
                            toneChunks.erase(pre_it);
                            ++processNum;
                        }
                    }
                }
            }
            ++it;
        }
        if (processNum == 0) {
            break;
        }
    }
}

inline auto getToneSection(const tonemap_t& tone_map,
                           const midiSearch::mu_melody_t& melody,
                           int sectionCount = 4,
                           int minSectionLen = 16) {
    midiSearch::mu_melody_t tmp;
    std::list<melodyChunk_t> arr;
    for (auto it : melody) {
        if (tmp.size() >= sectionCount) {
            arr.push_back(std::move(buildMelodyChunk(tone_map, tmp)));
            tmp.clear();
        }
        tmp.push_back(it);
    }
    getToneSection(tone_map, arr, minSectionLen);
    return arr;
}

}  // namespace chcpy::tone