#ifndef MGNR_SEQ2ID
#define MGNR_SEQ2ID
#include <set>
#include "cJSON.h"
#include "calcEditDist.h"
#include "filereader.h"
#include "generator.h"
#include "melody2chord.h"
#include "search.h"
namespace chcpy {
namespace seq2id {

template <typename T>  //要求必须有seq_id成员
concept dict_c = requires(T a) {
    a.seq_id;
    a.index;
};
using melody_t = midiSearch::melody_t;

struct dict_t {
    //seq_id是多对一关系，所以不支持反向转换
    std::map<melody_t, int> seq_id;
    int index = 0;
};

melody_t notePreprocess(const melody_t& seq) {
    melody_t res;
    std::map<int, int> count;
    std::vector<std::pair<int, int>> clist;
    for (auto it : seq) {
        if (it != -65535) {
            count[it]++;
        }
    }
    for (auto it : count) {
        clist.push_back(it);
    }
    sort(clist.begin(), clist.end(), [&](const std::pair<int, int>& A, const std::pair<int, int>& B) {
        if (A.second == B.second) {
            return A.first > B.first;
        } else {
            return A.second > B.second;
        }
    });
    for (auto it : clist) {
        res.push_back(it.first);
    }
    return res;
}

template <dict_c T>
inline int getIdBySeq(T& self, const melody_t& iseq) {
    auto seq = notePreprocess(iseq);
    auto it = self.seq_id.find(seq);
    if (it == self.seq_id.end()) {
        auto upper = self.seq_id.upper_bound(seq);
        if(upper!=self.seq_id.end()){
            return upper->second;
        }else{
            return 0;
        }
    } else {
        return it->second;
    }
}

template <dict_c T>
inline void load(T& self, const std::string& dic_path) {
    melody_t buf_seq;
    int buf_id;
    for (auto it : midiSearch::lineReader(dic_path)) {
        auto obj = cJSON_Parse(it->c_str());
        if (obj) {                            //解析json
            if (obj->type == cJSON_Object) {  //如果是对象
                bool haveId = false;
                bool haveSeq = false;
                auto line = obj->child;                      //获取头部指针
                while (line) {                               //开始遍历
                    if (strcmp(line->string, "val") == 0) {  //名称为val
                        if (line->type == cJSON_Number) {    //类型为数字
                            buf_id = line->valueint;         //设置id
                            haveId = true;                   //标记
                        }
                    } else if (strcmp(line->string, "key") == 0) {  //获取key
                        if (line->type == cJSON_Array) {            //类型为数组
                            buf_seq.clear();                        //初始化缓冲区
                            auto note = line->child;                //开始遍历
                            while (note) {
                                if (note->type == cJSON_Number) {       //类型为数字
                                    buf_seq.push_back(note->valueint);  //插入
                                    haveSeq = true;                     //标记
                                }
                                note = note->next;
                            }
                        }
                    }
                    line = line->next;
                }
                if (haveId && haveSeq) {
                    self.seq_id[buf_seq] = buf_id;
                    if (self.index < buf_id) {
                        self.index = buf_id;
                    }
                }
            }
            cJSON_Delete(obj);
        }
    }
}

template <dict_c T>
inline void save(T& self, const std::string& dic_path) {
    auto fp = fopen(dic_path.c_str(), "w");
    if (fp) {
        for (auto& it : self.seq_id) {
            auto obj = cJSON_CreateObject();
            cJSON_AddNumberToObject(obj, "val", it.second);
            auto notes = cJSON_CreateArray();
            cJSON_AddItemToObject(obj, "key", notes);
            for (auto note : it.first) {
                cJSON_AddItemToArray(notes, cJSON_CreateNumber(note));
            }
            auto s = cJSON_PrintUnformatted(obj);
            if (s) {
                fprintf(fp, "%s\n", s);
                free(s);
            }
            cJSON_Delete(obj);
        }
        fclose(fp);
    }
}

template <dict_c T>
inline int add(T& self, const melody_t& iseq) {
    auto seq = notePreprocess(iseq);
    auto it = self.seq_id.find(seq);
    if (it == self.seq_id.end()) {
        ++self.index;
        self.seq_id[seq] = self.index;
        return self.index;
    } else {
        auto& p = it->second;
        return p;
    }
}

inline midiSearch::generator<melody_t*> melody2seq(const melody_t& melody) {
    melody_t buf;
    for (auto it : melody) {
        buf.push_back(it);
        if (buf.size() >= 4) {
            co_yield &buf;
            buf.clear();
        }
    }
    if (!buf.empty()) {
        co_yield &buf;
        buf.clear();
    }
}

template <dict_c T>
inline void melody2seq(T& self, const melody_t& melody, melody_t& seq) {
    //每四个音符切分，获取id
    seq.clear();
    for (auto mel4 : melody2seq(melody)) {
        seq.push_back(getIdBySeq(self, *mel4));
    }
}

}  // namespace seq2id

namespace chord2id {

template <typename T>
concept dict_c = requires(T a) {
    a.chord_id;
    a.id_chord;
    a.index;
};

struct dict_t {
    std::map<std::string, int> chord_id;
    std::map<int, std::string> id_chord;
    int index = 0;
};

template <dict_c T>
inline int add(T& self, const std::string& ch) {
    auto it = self.chord_id.find(ch);
    if (it == self.chord_id.end()) {
        ++self.index;
        self.chord_id[ch] = self.index;
        self.id_chord[self.index] = ch;
        return self.index;
    } else {
        auto& p = it->second;
        return p;
    }
}

template <dict_c T>
constexpr int get(T& self, const std::string& ch) {
    auto it = self.chord_id.find(ch);
    if (it == self.chord_id.end()) {
        return -1;
    } else {
        auto& p = it->second;
        return p;
    }
}

template <dict_c T>
constexpr std::string get(T& self, int id) {
    auto it = self.id_chord.find(id);
    if (it == self.id_chord.end()) {
        return "";
    } else {
        auto& p = it->second;
        return p;
    }
}

template <dict_c T>
inline void save(T& self, const std::string& dic_path) {
    auto fp = fopen(dic_path.c_str(), "w");
    if (fp) {
        for (auto& it : self.chord_id) {
            auto obj = cJSON_CreateObject();
            cJSON_AddNumberToObject(obj, "val", it.second);
            cJSON_AddStringToObject(obj, "key", it.first.c_str());
            auto s = cJSON_PrintUnformatted(obj);
            if (s) {
                fprintf(fp, "%s\n", s);
                free(s);
            }
            cJSON_Delete(obj);
        }
        fclose(fp);
    }
}

template <dict_c T>
inline void load(T& self, const std::string& dic_path) {
    std::string buf_chord;
    int buf_id;
    for (auto it : midiSearch::lineReader(dic_path)) {
        auto obj = cJSON_Parse(it->c_str());
        if (obj) {                            //解析json
            if (obj->type == cJSON_Object) {  //如果是对象
                bool haveId = false;
                bool havechord = false;
                auto line = obj->child;                      //获取头部指针
                while (line) {                               //开始遍历
                    if (strcmp(line->string, "val") == 0) {  //名称为val
                        if (line->type == cJSON_Number) {    //类型为数字
                            buf_id = line->valueint;         //设置id
                            haveId = true;                   //标记
                        }
                    } else if (strcmp(line->string, "key") == 0) {  //获取key
                        if (line->type == cJSON_String) {           //类型为数组
                            buf_chord = line->valuestring;          //设置chord
                            havechord = true;                       //标记
                        }
                    }
                    line = line->next;
                }
                if (haveId && havechord) {
                    self.chord_id[buf_chord] = buf_id;
                    self.id_chord[buf_id] = buf_chord;
                    if (self.index < buf_id) {
                        self.index = buf_id;
                    }
                }
            }
            cJSON_Delete(obj);
        }
    }
}

}  // namespace chord2id
}  // namespace chcpy
#endif
