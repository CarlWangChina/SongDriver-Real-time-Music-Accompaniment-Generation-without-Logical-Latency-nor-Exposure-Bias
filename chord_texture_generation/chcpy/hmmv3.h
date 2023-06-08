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
#include "hmm.h"

namespace chcpy::hmm {

struct SMat3D_predict {
    inline SMat3D_predict(float v) {
        init(v);
    }
    inline SMat3D_predict() = default;
    std::map<std::tuple<short, short, short>, float> data{};
    float defaultValue = 0;
    inline float operator()(short x, short y, short z) const {
        auto it = data.find(std::make_tuple(x, y, z));
        if (it != data.end()) {
            return it->second;
        }
        return defaultValue;
    }
    inline void set(short x, short y, short z, float v) {
        data[std::make_tuple(x, y, z)] = v;
    }
    inline void add(short x, short y, short z, float v) {
        auto it = data.find(std::make_tuple(x, y, z));
        if (it != data.end()) {
            it->second += v;
        } else {
            data[std::make_tuple(x, y, z)] = defaultValue + v;
        }
    }
    inline void init(float def) {
        data.clear();
        defaultValue = def;
    }
};

struct SMat2D_predict {
    inline SMat2D_predict(float v) {
        init(v);
    }
    inline SMat2D_predict() = default;
    std::map<std::tuple<short, short>, float> data{};
    float defaultValue = 0;
    inline float operator()(short x, short y) const {
        auto it = data.find(std::make_tuple(x, y));
        if (it != data.end()) {
            return it->second;
        }
        return defaultValue;
    }
    inline void set(short x, short y, float v) {
        data[std::make_tuple(x, y)] = v;
    }
    inline void add(short x, short y, float v) {
        auto it = data.find(std::make_tuple(x, y));
        if (it != data.end()) {
            it->second += v;
        } else {
            data[std::make_tuple(x, y)] = defaultValue + v;
        }
    }
    inline void init(float def) {
        data.clear();
        defaultValue = def;
    }
};

struct SMat3D_train {
    std::map<short, std::map<short, std::map<short, float>>> data{};
    float defaultValue = 0;
    inline float operator()(short x, short y, short z) const {
        auto it_x = data.find(x);
        if (it_x != data.end()) {
            auto it_y = it_x->second.find(y);
            if (it_y != it_x->second.end()) {
                auto it_z = it_y->second.find(z);
                if (it_z != it_y->second.end()) {
                    return it_z->second;
                }
            }
        }
        return defaultValue;
    }
    inline void set(short x, short y, short z, float v) {
        data[x][y][z] = v;
    }
    inline void add(short x, short y, short z, float v) {
        auto it_x = data.find(x);
        if (it_x != data.end()) {
            auto it_y = it_x->second.find(y);
            if (it_y != it_x->second.end()) {
                auto it_z = it_y->second.find(z);
                if (it_z != it_y->second.end()) {
                    it_z->second += v;
                    return;
                }
            }
        }
        data[x][y][z] = defaultValue + v;
    }
    inline void init(float def) {
        data.clear();
        defaultValue = def;
    }
    inline void normalize() {
        for (auto& it_x : data) {
            for (auto& it_y : it_x.second) {
                float sum = 0.0;
                for (auto& it_z : it_y.second) {
                    sum += it_z.second;
                }
                if (sum == 0) {
                    for (auto& it_z : it_y.second) {
                        it_z.second = 0.0;
                    }
                } else {
                    for (auto& it_z : it_y.second) {
                        it_z.second /= sum;
                    }
                }
            }
        }
    }
};

struct SMat2D_train {
    std::map<short, std::map<short, float>> data{};
    float defaultValue = 0;
    inline float operator()(short x, short y) const {
        auto it_x = data.find(x);
        if (it_x != data.end()) {
            auto it_y = it_x->second.find(y);
            if (it_y != it_x->second.end()) {
                return it_y->second;
            }
        }
        return defaultValue;
    }
    inline void set(short x, short y, float v) {
        data[x][y] = v;
    }
    inline void add(short x, short y, float v) {
        auto it_x = data.find(x);
        if (it_x != data.end()) {
            auto it_y = it_x->second.find(y);
            if (it_y != it_x->second.end()) {
                it_y->second += v;
                return;
            }
        }
        data[x][y] = defaultValue + v;
    }
    inline void init(float def) {
        data.clear();
        defaultValue = def;
    }
    inline void normalize() {
        for (auto& it_x : data) {
            float sum = 0.0;
            for (auto& it_y : it_x.second) {
                sum += it_y.second;
            }
            if (sum == 0) {
                for (auto& it_y : it_x.second) {
                    it_y.second = 0.0;
                }
            } else {
                for (auto& it_y : it_x.second) {
                    it_y.second /= sum;
                }
            }
        }
    }
};

struct hmmv3_train_t {
    int32_t M;             //key数量
    int32_t N;             //val数量
    SMat2D_train A1;       //状态转移矩阵A[vid][vid]，格式[N,N]
    SMat3D_train A2;       //状态转移矩阵A[vid][vid][vid]，格式[N,N,N]
    SMat2D_train B1;       //发射矩阵，B[vid][kid]，格式[N,M]
    SMat3D_train B2;       //发射矩阵，B[vid][vid][kid]，格式[N,N,M]
    std::vector<float> P;  //初始概率P[vid]，格式[N]
};
template <typename T>
concept hmmv3_train_c = requires(T a) {
    a.A1;
    a.B1;
    a.A2;
    a.B2;
    a.P;
    a.M;
    a.N;
};

struct hmmv3_predict_t {
    int32_t M;                 //key数量
    int32_t N;                 //val数量
    SMat2D_predict A1_log;     //状态转移矩阵A[vid][vid]，格式[N,N]
    SMat3D_predict A2_log;     //状态转移矩阵A[vid][vid][vid]，格式[N,N,N]
    SMat2D_predict B1_log;     //发射矩阵，B[vid][kid]，格式[N,M]
    SMat3D_predict B2_log;     //发射矩阵，B[vid][vid][kid]，格式[N,N,M]
    std::vector<float> P_log;  //初始概率P[vid]，格式[N]
};
template <typename T>
concept hmmv3_predict_c = requires(T a) {
    a.A1_log;
    a.B1_log;
    a.A2_log;
    a.B2_log;
    a.P_log;
    a.M;
    a.N;
};

template <hmmv3_train_c T>
inline void save_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "w");
    if (fp) {
        fprintf(fp, "%d %d\n", self.M, self.N);
        //A1
        for (auto& it_x : self.A1.data) {
            for (auto& it_y : it_x.second) {
                float val = it_y.second;
                if (val != 0.0f) {
                    fprintf(fp, "A %d %d %f\n", it_x.first, it_y.first, val);
                }
            }
        }
        //A2
        for (auto& it_x : self.A2.data) {
            for (auto& it_y : it_x.second) {
                for (auto& it_z : it_y.second) {
                    float val = it_z.second;
                    if (val != 0.0f) {
                        fprintf(fp, "a %d %d %d %f\n", it_x.first, it_y.first, it_z.first, val);
                    }
                }
            }
        }
        //B1
        for (auto& it_x : self.B1.data) {
            for (auto& it_y : it_x.second) {
                float val = it_y.second;
                if (val != 0.0f) {
                    fprintf(fp, "B %d %d %f\n", it_x.first, it_y.first, val);
                }
            }
        }
        //B2
        for (auto& it_x : self.B2.data) {
            for (auto& it_y : it_x.second) {
                for (auto& it_z : it_y.second) {
                    float val = it_z.second;
                    if (val != 0.0f) {
                        fprintf(fp, "b %d %d %d %f\n", it_x.first, it_y.first, it_z.first, val);
                    }
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

template <hmmv3_train_c h>
inline void init(  //初始化
    h& self,
    int M,
    int N) {
    self.M = M;
    self.N = N;
    self.P = std::vector<float>(N, 0);
    self.A1.init(0);
    self.A2.init(0);
    self.B1.init(0);
    self.B2.init(0);
}

template <hmmv3_predict_c h>
inline void init(  //初始化
    h& self,
    int M,
    int N) {
    self.M = M;
    self.N = N;
    self.P_log = std::vector<float>(N, log_safe(0));
    self.A1_log.init(log_safe(0));
    self.A2_log.init(log_safe(0));
    self.B1_log.init(log_safe(0));
    self.B2_log.init(log_safe(0));
}

template <hmmv3_train_c T>
inline void load_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "r");
    if (fp) {
        char buf[128];
        bzero(buf, sizeof(buf));
        fgets(buf, sizeof(buf), fp);
        std::istringstream head(buf);
        int M, N;
        std::string mat;
        int x, y, z;
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
                self.A1.set(x, y, val);
            } else if (mat == "B") {
                line >> x;
                line >> y;
                line >> val;
                self.B1.set(x, y, val);
            } else if (mat == "a") {
                line >> x;
                line >> y;
                line >> z;
                line >> val;
                self.A2.set(x, y, z, val);
            } else if (mat == "b") {
                line >> x;
                line >> y;
                line >> z;
                line >> val;
                self.B2.set(x, y, z, val);
            } else if (mat == "P") {
                line >> x;
                line >> val;
                self.P.at(x) = val;
            }
        }
        fclose(fp);
    }
}

template <hmmv3_predict_c T>
inline void load_text(T& self, const std::string& path) {
    auto fp = fopen(path.c_str(), "r");
    if (fp) {
        char buf[128];
        bzero(buf, sizeof(buf));
        fgets(buf, sizeof(buf), fp);
        std::istringstream head(buf);
        int M, N;
        std::string mat;
        int x, y, z;
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
                self.A1_log.set(x, y, log_safe(val));
            } else if (mat == "B") {
                line >> x;
                line >> y;
                line >> val;
                self.B1_log.set(x, y, log_safe(val));
            } else if (mat == "a") {
                line >> x;
                line >> y;
                line >> z;
                line >> val;
                self.A2_log.set(x, y, z, log_safe(val));
            } else if (mat == "b") {
                line >> x;
                line >> y;
                line >> z;
                line >> val;
                self.B2_log.set(x, y, z, log_safe(val));
            } else if (mat == "P") {
                line >> x;
                line >> val;
                self.P_log.at(x) = log_safe(val);
            }
        }
        fclose(fp);
    }
}

template <hmmv3_train_c h>
inline void train_process(h& self, const std::function<void(std::pair<int, int>&)>& getData) {
    int prev1_tag = -1;
    int prev2_tag = -1;

    std::pair<int, int> line;
    while (1) {
        getData(line);
        int wordId = line.first;
        int tagId = line.second;
        if (wordId < 0 || tagId < 0 || wordId >= self.M || tagId >= self.N) {
            break;
        }
        if (prev2_tag < 0 || prev2_tag >= self.N) {
            self.P[tagId] += 1;
            self.B1.add(tagId, wordId, 1);
            self.B2.add(tagId, tagId, wordId, 1);
        } else if (prev1_tag < 0 || prev1_tag >= self.N) {
            self.A1.add(prev1_tag, tagId, 1);
            self.B1.add(tagId, wordId, 1);
        } else {
            self.A1.add(prev1_tag, tagId, 1);
            self.A2.add(prev2_tag, prev1_tag, tagId, 1);
            self.B1.add(tagId, wordId, 1);
            self.B2.add(prev1_tag, tagId, wordId, 1);
        }
        prev2_tag = prev1_tag;
        prev1_tag = tagId;
    }
}

template <hmmv3_train_c h>
inline void train_end(h& self) {
    normalize(self.P);
    self.A1.normalize();
    self.A2.normalize();
    self.B1.normalize();
    self.B2.normalize();
}

//此方法很慢，慎用
template <hmmv3_predict_c h>
inline void predict(                 //维特比算法，获得最优切分路径
    const h& self,                   //hmm对象
    const melody_t& seq,             //seq须先用melody2seq预处理
    std::vector<int>& best_sequence  //输出
) {
    auto T = seq.size();
    constexpr auto log0 = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::vector<float>>> dp(T, std::vector<std::vector<float>>(self.N, std::vector<float>(self.N, log0)));
    std::vector<std::vector<std::vector<float>>> ptr(T, std::vector<std::vector<float>>(self.N, std::vector<float>(self.N, 0)));

#pragma omp parallel for
    for (int x = 0; x < self.N; x++) {
        for (int y = 0; y < self.N; y++) {
            float val = self.P_log.at(x) + self.B1_log(x, seq.at(0)) +
                        self.A1_log(x, y) + self.B2_log(x, y, seq.at(1));
            dp[1][x][y] = val;
            ptr[1][x][y] = -1;
        }
    }

    for (int i = 2; i < T; i++) {
        int num_3 = seq.at(i);
        auto& dpi = dp.at(i);
        auto& dp_last = dp.at(i - 1);
#pragma omp parallel for
        for (int j = 0; j < self.N; j++) {
#pragma omp parallel for
            for (int k1 = 0; k1 < self.N; k1++) {
                float max_val = log0;
                int max_path = -1;
                for (int k2 = 0; k2 < self.N; k2++) {
                    float val = dp_last[k2][j] + self.A2_log(k2, j, k1);
                    if (val > max_val) {
                        max_val = val;
                        max_path = k2;
                    }
                }
                dpi[j][k1] = max_val + self.B2_log(j, k1, num_3);
                ptr[i][j][k1] = max_path;
            }
        }
    }

    best_sequence.resize(T);
    //argmax 2d
    float max_val = log0;
    int max_path_i = -1;
    int max_path_j = -1;
    auto& dpi = dp.at(T - 1);
    for (int i = 0; i < self.N; i++) {
        for (int j = 0; j < self.N; j++) {
            if (dpi[i][j] > max_val) {
                max_val = dpi[i][j];
                max_path_i = i;
                max_path_j = j;
            }
        }
    }
    best_sequence.at(T - 1) = max_path_j;
    best_sequence.at(T - 2) = max_path_i;

    for (int t = T - 1; t > 1; t--) {
        int max_path_k = ptr[t][max_path_i][max_path_j];
        best_sequence.at(t - 2) = max_path_k;
        max_path_j = max_path_i;
        max_path_i = max_path_k;
    }
}

}  // namespace chcpy::hmm