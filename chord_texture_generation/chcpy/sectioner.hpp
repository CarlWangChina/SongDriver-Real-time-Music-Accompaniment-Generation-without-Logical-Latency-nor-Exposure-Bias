#ifndef MIDILIB_SECTIONER_HPP
#define MIDILIB_SECTIONER_HPP
#include <array>
#include <cmath>
#include <list>
#include <map>
#include <vector>

namespace autochord {

struct sectioner {
    std::list<int> sections;
    int noteNum;
};

template <typename T>
void sectioner_init(T& self, int num) {
    printf("chordGen:sectioner_init\n");
    self.noteNum = num;
    self.sections.clear();
    printf("chordGen:sectioner_init success\n");
}

template <typename T>
void sectioner_pushNote(T& self, int note) {
    self.sections.push_back(note);
    while (self.sections.size() > self.noteNum) {
        self.sections.pop_front();
    }
}

inline int sectioner_getNoteFromArray(const std::vector<int>& arr) {
    std::map<int, int> counts;
    //统计
    for (auto it : arr) {
        if (it > 0) {
            ++counts[it];
        }
    }

    int last = -2;  //上一个音符

    //局部
    int maxNum_local = 0;
    int maxNum_Note_local = -2;
    int noteNum_local = 0;

    //全局
    int maxNum = 0;
    int maxNum_Note = -2;
    int noteNum = 0;

    for (auto& it : counts) {
        int nowNote = it.first;      //当前的音符
        int nowNoteNum = it.second;  //当前音符的数量
        noteNum += nowNoteNum;       //全局计数器
        if (nowNote - last == 1) {
            //和上一个相差1，说明在同一个tuple
            noteNum_local += nowNoteNum;  //加入局部总数
            //判断是否为局部最大
            if (nowNoteNum > maxNum_local) {
                maxNum_local = nowNoteNum;
                maxNum_Note_local = nowNote;
            }
        } else {
            //不在同一个tuple
            //设置全局最大值
            if (noteNum_local > maxNum) {  //局部的所有音符都算在最大的上面，所以用noteNum_local
                maxNum = noteNum_local;
                maxNum_Note = maxNum_Note_local;
            }
            //清空局部记录(把自己作为第一个)
            maxNum_local = nowNoteNum;
            noteNum_local = nowNoteNum;
            maxNum_Note_local = nowNote;
        }
        last = nowNote;
    }
    //收尾工作
    if (noteNum_local > maxNum) {
        maxNum = noteNum_local;
        maxNum_Note = maxNum_Note_local;
    }

    if (maxNum_Note <= 0 || noteNum <= 0) {
        return 0;
    }
    float p = ((float)maxNum) / ((float)noteNum);
    if (p > 0.6) {  //占比超过60%才算
        return maxNum_Note;
    } else {
        return 0;
    }
}

template <typename T>
void sectioner_pushNoteFromArray(T& self, const std::vector<int>& arr) {
    int n = sectioner_getNoteFromArray(arr);
    sectioner_pushNote(self, n);
}

template <typename T>
float sectioner_getVariance(T& self) {
    //计算方差
    int sum = 0;
    int count = 0;
    for (auto it : self.sections) {
        if (it > 0) {  //音符不为0才计入
            sum += it;
            ++count;
        }
    }
    if (count == 0)
        return 0;
    float average = ((float)sum) / ((float)count);  //平均数

    //方差
    float delta2Sum = 0;
    for (auto it : self.sections) {
        if (it > 0) {
            float delta = it - average;
            delta2Sum += delta * delta;
        }
    }
    return delta2Sum / count;
}

template <typename T>
float sectioner_getAverDelta(T& self) {
    //计算平均变化率
    int last = -1;
    int sum = 0;
    int count = 0;
    for (auto it : self.sections) {
        if (it > 0) {  //音符不为0才计入
            if (last != -1) {
                int delta = std::abs(it - last);
                if (delta >= 12) {
                    continue;
                }
                sum += delta;
                ++count;
            }
            last = it;
        } else {
            //没有音符写入0占位
            if (last != -1) {
                ++count;
            }
        }
    }
    if (count == 0) {
        return 0;
    } else {
        return ((float)sum) / ((float)count);
    }
}

}  // namespace autochord
#endif  //MIDILIB_SECTIONER_HPP
