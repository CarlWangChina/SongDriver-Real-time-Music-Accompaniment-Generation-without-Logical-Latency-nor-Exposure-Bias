#pragma once
#include <memory.h>
#include <stdio.h>
#include <array>
#include <map>
#include <sstream>
#include <vector>
namespace chcpy::bayes {

template <int probnum = 5>
struct bayes_predict_t {
    std::array<std::map<int, std::vector<std::pair<int, float>>>, probnum> prob{};

    inline std::vector<std::pair<int, float>> getProb(const std::array<int, probnum>& input) const {
#ifdef CHCPY_DEBUG
        clock_t startTime, endTime;
        startTime = clock();  //计时开始
#endif
        bool first = true;

        //双缓冲
        std::vector<std::pair<int, float>> res_1;
        std::vector<std::pair<int, float>> res_2;
        auto res = &res_1;
        auto res_back = &res_2;

        for (size_t argIndex = 0; argIndex < probnum; ++argIndex) {
            auto& prob_i = prob[argIndex];
            auto prob_it = prob_i.find(input[argIndex]);
            if (prob_it != prob_i.end()) {
                if (first) {
                    *res = prob_it->second;
                } else {
                    res_back->clear();
                    /*
                    * 代码性能过低，已废弃
                    if (prob_it->second.size() > res->size()) {
                        for (auto p : *res) {
                            //取交集
                            auto it = prob_it->second.find(p.first);
                            if (it != prob_it->second.end()) {
                                (*res_back)[p.first] = p.second * it->second;
                            }
                        }
                    } else {
                        for (auto p : prob_it->second) {
                            //取交集
                            auto it = res->find(p.first);
                            if (it != res->end()) {
                                (*res_back)[p.first] = p.second * it->second;
                            }
                        }
                    }
                    */
                    auto& ptr1 = prob_it->second;
                    auto& ptr2 = *res;
                    int len1 = ptr1.size(), len2 = ptr2.size();
                    int i = 0, j = 0;
                    while (i < len1 && j < len2) {
                        if (ptr1[i].first == ptr2[j].first) {
                            res_back->push_back(std::pair<int, float>(
                                ptr1[i].first,
                                ptr1[i].second * ptr2[j].second));
                            i++;
                            j++;
                        } else if (ptr1[i].first > ptr2[j].first) {
                            j++;
                        } else {
                            i++;
                        }
                    }

                    auto tmp = res_back;
                    res_back = res;
                    res = tmp;
                }
                first = false;
            }
        }
#ifdef CHCPY_DEBUG
        endTime = clock();  //计时结束
        printf("\nbayes用时%f秒\n", (float)(endTime - startTime) / CLOCKS_PER_SEC);
#endif
        return *res;
    }
    inline void load(const char* path) {
        auto fp = fopen(path, "r");
        if (fp) {
            char buf[128];
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            while (!feof(fp)) {
                bzero(buf, sizeof(buf));
                fgets(buf, sizeof(buf), fp);
                std::istringstream line(buf);
                int i = -1, j = -1, k = -1;
                float v = 0;
                line >> i;
                line >> j;
                line >> k;
                line >> v;
                if (i >= 0 && i < probnum) {
                    prob.at(i)[j].push_back(std::pair<int, float>(k, v));
                }
            }
            for (auto& it_i : prob) {
                for (auto& it_j : it_i) {
                    std::sort(
                        it_j.second.begin(),
                        it_j.second.end(),
                        [](const std::pair<int, float>& A,
                           const std::pair<int, float>& B) {
                            return A.first < B.first;
                        });
                }
            }
            fclose(fp);
        }
    }
};

template <int probnum = 5>
struct bayes_train_t {
    std::array<std::map<int, std::map<int, float>>, probnum> count_c;
    std::array<std::map<int, float>, probnum> count_e;
    inline void add(const std::array<int, probnum>& input, int val) {
        for (size_t i = 0; i < probnum; ++i) {
            count_c[i][input[i]][val] += 1;
            count_e[i][input[i]] += 1;
        }
    }
    inline void normalize() {
        for (size_t i = 0; i < probnum; ++i) {
            auto& count_c_i = count_c[i];
            auto& count_e_i = count_e[i];
            float sum_e = 0;
            for (auto& it : count_e_i) {
                sum_e += it.second;
            }
            for (auto& it : count_e_i) {
                it.second /= sum_e;
            }
            for (auto& it_c : count_c_i) {
                float sum_c = 0;
                for (auto& it : it_c.second) {
                    sum_c += it.second;
                }
                auto it_e = count_e_i.find(it_c.first);
                if (it_e == count_e_i.end()) {
                    printf("找不到元素：i=%d e=%d\n", i, it_c.first);
                } else {
                    for (auto& it : it_c.second) {
                        //printf("归一化：i=%d e=%d v=%f c=%f sum=%f\n",
                        //       i, it_c.first, it_e->second, it.second, sum_c);
                        it.second /= (sum_c * it_e->second);
                    }
                }
            }
        }
    }
    inline void save(const char* path) {
        auto fp = fopen(path, "w");
        if (fp) {
            for (size_t i = 0; i < probnum; ++i) {
                auto& count_c_i = count_c[i];
                for (auto& it_i : count_c_i) {
                    for (auto& it_v : it_i.second) {
                        fprintf(fp, "%d %d %d %f\n", i, it_i.first, it_v.first, it_v.second);
                    }
                }
            }
            fclose(fp);
        }
    }
};

}  // namespace chcpy::bayes