#pragma once
#include <math.h>
#include <memory.h>
#include <omp.h>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace chcpy::hmm {

using melody_t = std::vector<int>;
template <typename T>
concept hmm_train_c = requires(T a) {
    a.A;
    a.B;
    a.P;
    a.M;
    a.N;
};
struct hmm_train_t {
    int32_t M;                          //key数量
    int32_t N;                          //val数量
    std::vector<std::vector<float>> A;  //状态转移矩阵A[vid][vid]，格式[N,N]
    std::vector<std::vector<float>> B;  //发射矩阵，B[vid][kid]，格式[N,M]
    std::vector<float> P;               //初始概率P[vid]，格式[N]
};
template <typename T>
concept hmm_predict_c = requires(T a) {
    a.A_log;
    a.B_log;
    a.P_log;
    a.M;
    a.N;
};
struct hmm_predict_t {
    int32_t M;                              //key数量
    int32_t N;                              //val数量
    std::vector<std::vector<float>> A_log;  //状态转移矩阵A[vid][vid]，格式[N,N]
    std::vector<std::vector<float>> B_log;  //发射矩阵，B[vid][kid]，格式[N,M]
    std::vector<float> P_log;               //初始概率P[vid]，格式[N]
};

template <hmm_train_c T>
inline void save_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "w");
    if (fp) {
        fprintf(fp, "%d %d\n", self.M, self.N);
        //A
        for (int i = 0; i < self.N; ++i) {
            for (int j = 0; j < self.N; ++j) {
                float val = self.A.at(i).at(j);
                if (val != 0.0f) {
                    fprintf(fp, "A %d %d %f\n", i, j, val);
                }
            }
        }
        //B
        for (int i = 0; i < self.N; ++i) {
            for (int j = 0; j < self.M; ++j) {
                float val = self.B.at(i).at(j);
                if (val != 0.0f) {
                    fprintf(fp, "B %d %d %f\n", i, j, val);
                }
            }
        }
        //P
        for (int i = 0; i < self.N; ++i) {
            float val = self.P.at(i);
            if (val != 0.0f) {
                fprintf(fp, "P %d %f\n", i, val);
            }
        }
        fclose(fp);
    }
}

template <hmm_train_c T>
inline void load_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "r");
    if (fp) {
        char buf[128];
        bzero(buf, sizeof(buf));
        fgets(buf, sizeof(buf), fp);
        std::istringstream head(buf);
        int M, N;
        std::string mat;
        int x, y;
        float val;
        head >> M;
        head >> N;
        init(self, M, N);
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            std::istringstream line(buf);
            line >> mat;
            if (mat == "A") {
                line >> x;
                line >> y;
                line >> val;
                self.A.at(x).at(y) = val;
            } else if (mat == "B") {
                line >> x;
                line >> y;
                line >> val;
                self.B.at(x).at(y) = val;
            } else if (mat == "P") {
                line >> x;
                line >> val;
                self.P.at(x) = val;
            }
        }
        fclose(fp);
    }
}

template <class ForwardIterator>
constexpr size_t argmin(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::min_element(first, last));
}

template <class ForwardIterator>
constexpr size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

constexpr float log_safe(float v) {
    if (v == 0) {
        return -std::numeric_limits<float>::infinity();
    }
    return log(v);
}

template <hmm_train_c h>
inline void init(  //初始化
    h& self,
    int M,
    int N) {
    self.M = M;
    self.N = N;
    self.P = std::vector<float>(N, 0);
    self.B = std::vector<std::vector<float>>(N, std::vector<float>(M, 0));
    self.A = std::vector<std::vector<float>>(N, std::vector<float>(N, 0));
}

template <hmm_predict_c h>
inline void init(  //初始化
    h& self,
    int M,
    int N) {
    self.M = M;
    self.N = N;
    self.P_log = std::vector<float>(N, log_safe(0));
    self.B_log = std::vector<std::vector<float>>(N, std::vector<float>(M, log_safe(0)));
    self.A_log = std::vector<std::vector<float>>(N, std::vector<float>(N, log_safe(0)));
}

template <hmm_predict_c T>
inline void load_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "r");
    if (fp) {
        char buf[128];
        bzero(buf, sizeof(buf));
        fgets(buf, sizeof(buf), fp);
        std::istringstream head(buf);
        int M, N;
        std::string mat;
        int x, y;
        float val;
        head >> M;
        head >> N;
        init(self, M, N);
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            std::istringstream line(buf);
            line >> mat;
            if (mat == "A") {
                line >> x;
                line >> y;
                line >> val;
                self.A_log.at(x).at(y) = log_safe(val);
            } else if (mat == "B") {
                line >> x;
                line >> y;
                line >> val;
                self.B_log.at(x).at(y) = log_safe(val);
            } else if (mat == "P") {
                line >> x;
                line >> val;
                self.P_log.at(x) = log_safe(val);
            }
        }
        fclose(fp);
    }
}

template <hmm_train_c h>
inline void train_process(h& self, const std::function<void(std::pair<int, int>&)>& getData) {
    int prev_tag = -1;
    std::pair<int, int> line;
    while (1) {
        getData(line);
        int wordId = line.first;
        int tagId = line.second;
        if (wordId < 0 || tagId < 0 || wordId >= self.M || tagId >= self.N) {
            break;
        }
        if (prev_tag < 0 || prev_tag >= self.N) {
            self.P[tagId] += 1;
            self.B[tagId][wordId] += 1;
        } else {
            self.B[tagId][wordId] += 1;
            self.A[prev_tag][tagId] += 1;
        }
        prev_tag = tagId;
    }
}

template <typename T>
inline void normalize(T& a) {
    auto sum = std::accumulate(a.begin(), a.end(), 0);
    if (sum != 0) {
        for (auto& el : a) {
            el = el / sum;
        }
    } else {
        for (auto& el : a) {
            el = 0;
        }
    }
}

template <hmm_train_c h>
inline void train_end(h& self) {
    //归一化得到概率
    normalize(self.P);
    for (auto& it : self.A) {
        normalize(it);
    }
    for (auto& it : self.B) {
        normalize(it);
    }
}

template <hmm_predict_c h>
inline void predict(                 //维特比算法，获得最优切分路径
    const h& self,                   //hmm对象
    const melody_t& seq,             //seq须先用melody2seq预处理
    std::vector<int>& best_sequence  //输出
) {
#ifdef CHCPY_DEBUG
    clock_t startTime, endTime;
    startTime = clock();  //计时开始
#endif
    auto T = seq.size();
    constexpr auto log0 = -std::numeric_limits<float>::infinity();
    std::vector<std::vector<float>> dp(T, std::vector<float>(self.N, 0));
    std::vector<std::vector<int>> ptr(T, std::vector<int>(self.N, 0));

    std::vector<int> lastAvaible;

    std::mutex locker;

#pragma omp parallel for
    for (size_t j = 0; j < self.N; ++j) {
        float startval = self.P_log.at(j) + self.B_log.at(j).at(seq.at(0));
        dp.at(0).at(j) = startval;
        if (startval != log0) {
            locker.lock();
            lastAvaible.push_back(j);
            locker.unlock();
        }
    }
    for (size_t i = 1; i < T; ++i) {
        auto& dpi = dp.at(i);
        auto& dp_last = dp.at(i - 1);
#pragma omp parallel for
        for (size_t j = 0; j < self.N; ++j) {
            dpi.at(j) = log0;
            float base = self.B_log.at(j).at(seq.at(i));
            //for (size_t k = 0; k < self.N; ++k) {
            for (auto k : lastAvaible) {
                float score = base + dp_last.at(k) + self.A_log.at(k).at(j);
                if (score != log0 && score > dpi.at(j)) {
                    dpi.at(j) = score;
                    ptr.at(i).at(j) = k;
                }
            }
        }
        lastAvaible.clear();
#pragma omp parallel for
        for (size_t k = 0; k < self.N; ++k) {
            if (dpi.at(k) != log0) {
                locker.lock();
                lastAvaible.push_back(k);
                locker.unlock();
            }
        }
    }
    best_sequence.resize(T);
    best_sequence.at(T - 1) = argmax(dp.at(T - 1).begin(), dp.at(T - 1).end());
    for (int i = T - 2; i >= 0; --i) {  //这里不能用size_t，否则将导致下溢出，造成死循环
        best_sequence.at(i) = ptr.at(i + 1).at(best_sequence.at(i + 1));
    }
#ifdef CHCPY_DEBUG
    endTime = clock();  //计时结束
    printf("\nhmm用时%f秒\n", (float)(endTime - startTime) / CLOCKS_PER_SEC);
#endif
}

}  // namespace chcpy::hmm