#ifndef AUTOCHORD_CRF
#define AUTOCHORD_CRF
#include <memory.h>
#include <algorithm>
#include <array>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
namespace autochord {

const char* section_Map[] = {"1", "2", "3", "4", "5", "6", "7", "8"};

struct chordPredictor_t {
    std::array<std::unordered_map<std::string, int>, 14> T;
    std::vector<double> prob;
    std::vector<std::string> outName;
    std::vector<const char*> pushT_buffer;
    std::vector<std::tuple<std::string, double> > probOut;  //计算结果
    std::vector<std::string> vec;
    std::vector<double> prob_buffer;
};

template <typename tp>
void chordPredictor_pushT(tp& self, char* buf) {  //buf可能会被修改，目的是节省内存
    self.pushT_buffer.clear();
    self.pushT_buffer.push_back(buf);
    char* pbuf = buf;
    while (*pbuf) {
        if (*pbuf == ' ' || *pbuf == ':') {
            *pbuf = '\0';  //切断字符串
            char next = *(pbuf + 1);
            if (next != ' ' || next != ':' || next != '\0') {
                self.pushT_buffer.push_back(pbuf + 1);
            }
        }
        ++pbuf;
    }
    if (self.pushT_buffer.size() >= 3) {
        int id = atoi(self.pushT_buffer[0]);
        int inp_pos = atoi(self.pushT_buffer[1] + 1);
        auto inp_name = self.pushT_buffer[2];
        self.T[inp_pos][inp_name] = id;
    }
}

template <typename tp>
void chordPredictor_pushProb(tp& self, double prob) {
    self.prob.push_back(prob);
}

template <typename tp>
void chordPredictor_pushOutName(tp& self, const char* buf) {
    self.outName.push_back(buf);
}

inline void removeBr(char* buf) {
    char* pbuf = buf;
    while (*pbuf) {
        if (*pbuf == '\n') {
            *pbuf = '\0';
            return;
        }
        ++pbuf;
    }
}

template <typename tp>
void chordPredictor_loadModel(tp& self, const char* path) {
    FILE* fp = fopen(path, "r");
    char buf[1024];
    if (fp) {
        printf("crf:loading model %s\n", path);
        int count = 0;
        self.prob.clear();
        for (int i = 0; i < 14; ++i) {
            self.T[i].clear();
        }
        self.outName.clear();
        //处理头部
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            ++count;
            if (buf[0] == '\n') {  //空行
                break;
            }
        }
        printf("crf:read header:%d\n", count);
        count = 0;
        //setOutName
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            if (buf[0] == '\n') {  //空行
                break;
            } else {
                ++count;
                removeBr(buf);
                chordPredictor_pushOutName(self, buf);
            }
        }
        printf("crf:read out name:%d\n", count);
        count = 0;
        //丢弃模型参数
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            if (buf[0] == '\n') {  //空行
                break;
            } else {
                ++count;
            }
        }
        printf("crf:read template:%d\n", count);
        count = 0;
        //setT
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            if (buf[0] == '\n') {  //空行
                break;
            } else {
                removeBr(buf);
                chordPredictor_pushT(self, buf);
                ++count;
            }
        }
        printf("crf:read T:%d\n", count);
        count = 0;
        //setProb
        while (!feof(fp)) {
            bzero(buf, sizeof(buf));
            fgets(buf, sizeof(buf), fp);
            if (buf[0] == '\n') {  //空行
                break;
            } else {
                removeBr(buf);
                chordPredictor_pushProb(self, atof(buf));
                ++count;
            }
        }
        printf("crf:read prob:%d\n", count);
        fclose(fp);
    } else {
        printf("crf:load model fail\n");
    }
}

template <typename tp>
void chordPredictor_getProb(tp& self,
                            const std::list<std::string>& melody,
                            const std::list<std::string>& chord,
                            int id) {
    self.vec.clear();
    if (melody.size() + chord.size() + 2 != self.T.size()) {
        printf("length error\n");
        return;
    }
    for (auto it : melody) {
        self.vec.push_back(it);
    }
    self.vec.push_back("0");
    for (auto it : chord) {
        self.vec.push_back(it);
    }
    if (id >= 0) {
        self.vec.push_back(section_Map[id % 8]);
    } else {
        self.vec.push_back("0");
    }
    int outNameNum = self.outName.size();
    self.prob_buffer.resize(outNameNum);
    for (int i = 0; i < outNameNum; ++i) {
        self.prob_buffer[i] = 0;
    }

    int len = self.vec.size();
    for (int i = 0; i < len; ++i) {
        try {
            if (!self.vec[i].empty()) {
                auto idit = self.T[i].find(self.vec[i]);
                if (idit != self.T[i].end()) {
                    int id = idit->second;
                    for (int j = 0; j < outNameNum; ++j) {
                        double p = self.prob.at(id + j);
                        self.prob_buffer[j] += p;
                    }
                }
            }
        } catch (...) {
        }
    }

    self.probOut.clear();
    for (int i = 0; i < outNameNum; ++i) {
        self.probOut.push_back(std::make_tuple(self.outName[i], self.prob_buffer[i]));
    }

    std::sort(self.probOut.begin(), self.probOut.end(),
              [](std::tuple<std::string, double> a, std::tuple<std::string, double> b) {
                  return std::get<1>(a) > std::get<1>(b);
              });
}

}  // namespace autochord
#endif