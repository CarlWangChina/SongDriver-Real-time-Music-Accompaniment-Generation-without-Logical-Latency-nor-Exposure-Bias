#pragma once
#include <omp.h>
#include <string.h>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include "calcEditDist.h"

namespace mgnr::melody2chord {

using wmelody_t = std::vector<std::tuple<int, float>>;  //加权旋律

template <typename C>
inline void getComp(                                   //求组合
    const std::vector<std::tuple<int, float>>& notes,  //输入已排序的集合{(音符，音符的数量),...}
    C callback                                         //回调函数(音符集合,需要改变的数量)
) {
    //求音符总数
    float sum = 0;
    for (auto it : notes) {
        sum += std::get<1>(it);
    }
    //开始求组合
    int len = notes.size();
    std::vector<std::tuple<int, float>> res;
    for (int i = 0; i < len; ++i) {
        res.clear();
        float compCount = 0;
        for (int j = i; j < len; ++j) {
            auto it = notes.at(j);
            res.push_back(it);
            compCount += std::get<1>(it);
        }
        callback(res, sum - compCount);
    }
}
float calcEditDist(const std::vector<std::tuple<int, float>>& A, const std::vector<int>& B, float act = 1.0) {
    int lenA = A.size();
    int lenB = B.size();
    int i, j;
    std::vector<std::vector<float>> dp(lenA + 1, std::vector<float>(lenB + 1));

    //初始化边界
    for (i = 1; i <= lenA; i++) {
        dp.at(i).at(0) = std::get<1>(A.at(i - 1)) + dp.at(i - 1).at(0);
    }
    for (j = 1; j <= lenB; j++) {
        dp.at(0).at(j) = j;
    }

    //dp
    for (i = 1; i <= lenA; i++) {
        for (j = 1; j <= lenB; j++) {
            //if (s1[i] == s2[j])//运行结果不对，因为第i个字符的索引为i-1
            if (std::get<0>(A.at(i - 1)) == B.at(j - 1)) {
                dp.at(i).at(j) = dp.at(i - 1).at(j - 1);  //第i行j列的步骤数等于第i-1行j-1列，因为字符相同不需什么操作，所以不用+act
            } else {
                float weight_last = std::get<1>(A.at(i - 1));
                float weight_now = i < lenA ? std::get<1>(A.at(i)) : act;
                dp.at(i).at(j) = std::min(std::min(
                                              dp.at(i - 1).at(j) + weight_last,
                                              dp.at(i).at(j - 1) + weight_now),
                                          dp.at(i - 1).at(j - 1) + weight_last) +
                                 act;  //+act表示经过了一次操作
            }
        }
    }
#ifdef MGNR_DEBUG
    printf("lenA=%d lenB=%d\n", lenA, lenB);
    for (i = 0; i <= lenA; i++) {
        for (j = 0; j <= lenB; j++) {
            printf("%f\t", dp[i][j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    return dp.at(lenA).at(lenB);
}

inline void notes2RelativeChord(
    const std::vector<std::tuple<int, float>>& notes,
    std::vector<std::tuple<int, float>>& relativeChord) {
    //构造数字形式的和弦名称
    bool first = true;
    int firstNote = std::get<0>(notes.at(0));
    relativeChord.clear();
    for (auto it : notes) {
        relativeChord.push_back(std::make_tuple(std::get<0>(it) - firstNote, std::get<1>(it)));
    }
}

inline int getNumberNoteId(const char* note) {  //此函数性能低下，请勿在初始化以外的地方使用
    static const char* noteMap_1[] = {"1", "1#", "2", "2#", "3", "4", "4#", "5", "5#", "6", "6#", "7"};
    static const char* noteMap_2[] = {"1", "2b", "2", "3b", "3", "4", "5b", "5", "6b", "6", "7b", "7"};
    static const char* noteMap_3[] = {"1", "#1", "2", "#2", "3", "4", "#4", "5", "#5", "6", "#6", "7"};
    static const char* noteMap_4[] = {"1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"};
    for (int i = 0; i < 12; ++i) {
        if (
            strcmp(note, noteMap_1[i]) == 0 ||
            strcmp(note, noteMap_2[i]) == 0 ||
            strcmp(note, noteMap_3[i]) == 0 ||
            strcmp(note, noteMap_4[i]) == 0) {
            return i;
        }
    }
    return 0;
}
inline std::vector<int> getChordArray(const std::vector<const char*>& arr) {
    std::vector<int> res;
    for (auto it : arr) {
        int note = getNumberNoteId(it);
        res.push_back(note);
    }
    return res;
}

template <typename T>  //要求必须有chord_map成员
concept chord_map_c = requires(T a) {
    a.chord_map;
};

struct chordMap {
    std::map<std::string, std::vector<int>> chord_map;
    inline void addChord(const char* name, const std::vector<const char*>& chord) {
        auto ch = getChordArray(chord);
#ifdef MGNR_DEBUG
        printf("%s\t", name);
        for (auto it : ch) {
            printf("%d\t", it);
        }
        printf("\n");
#endif
        chord_map[name] = ch;
    }
    inline chordMap() {
        //addChord("single",  {"0"});
        addChord("M3", {"1", "3", "5"});
        addChord("m3", {"1", "b3", "5"});
        addChord("-5", {"1", "3", "b5"});
        addChord("ang", {"1", "3", "#5"});
        addChord("dim", {"1", "b3", "b5", "6"});
        addChord("sus", {"1", "4", "5"});
        addChord("6", {"1", "3", "5", "6"});
        addChord("m6", {"1", "b3", "5", "6"});
        addChord("7", {"1", "3", "5", "b7"});
        addChord("maj7", {"1", "3", "5", "7"});
        addChord("m7", {"1", "b3", "5", "b7"});
        addChord("m#7", {"1", "b3", "5", "7"});
        addChord("7+5", {"1", "3", "#5", "b7"});
        addChord("7-5", {"1", "3", "b5", "b7"});
        addChord("m7-5", {"1", "b3", "b5", "b7"});
        addChord("7sus4", {"1", "3", "5", "b7", "4"});
        addChord("7/6", {"1", "3", "5", "b7", "6"});
        addChord("maj9", {"1", "3", "5", "7", "2"});
        addChord("9", {"1", "3", "5", "b7", "2"});
        addChord("9+5", {"1", "3", "#5", "b7", "2"});
        addChord("9-5", {"1", "3", "b5", "b7", "2"});
        addChord("m9", {"1", "b3", "5", "b7", "2"});
        addChord("7+9", {"1", "3", "5", "b7", "#2"});
        addChord("m9#7", {"1", "b3", "5", "b7", "2"});
        addChord("7b9", {"1", "3", "5", "b7", "b2"});
        addChord("7-9+5", {"1", "3", "#5", "b7", "b2"});
        addChord("7-9-5", {"1", "3", "b5", "b7", "b2"});
        addChord("69", {"1", "3", "5", "6", "2"});
        addChord("m69", {"1", "b3", "5", "6", "2"});
        addChord("11", {"1", "3", "5", "b7", "2", "4"});
        addChord("m11", {"1", "b3", "5", "b7", "2", "4"});
        addChord("11+", {"1", "3", "5", "b7", "2", "#4"});
        addChord("13", {"1", "3", "5", "b7", "2", "4", "6"});
        addChord("13-9", {"1", "3", "5", "b7", "b2", "4", "6"});
        addChord("13-9-5", {"1", "3", "b5", "b7", "b2", "4", "6"});
    }
};

template <chord_map_c T>
inline std::tuple<int, std::string, int, float> noteCount2Chord(const T& self, const std::vector<std::tuple<int, float>>& notes, float act = 1.0) {
    if (notes.empty()) {
        return std::make_tuple(0, "_", 0, 0);
    }
    std::string min_chord;
    float min_base;
    float min_chord_weight;
    int min_chord_inverted = 0;
    bool first = true;
    getComp(notes, [&](const std::vector<std::tuple<int, float>>& n, float baseWeight) {
        std::vector<std::tuple<int, float>> relative;
        notes2RelativeChord(n, relative);
#ifdef MGNR_DEBUG
        printf("note:");
        for (auto it : relative) {
            printf("%d:%f ", std::get<0>(it), std::get<1>(it));
        }
        printf("\n");
#endif
        for (auto& it : self.chord_map) {
            ////检查基音是否符合
            const static bool isNotHalf[] = {true, false, true, false, true, true, false, true, false, true, false, true};
            for (auto chord_map_e : it.second) {
                if (!isNotHalf[(chord_map_e + std::get<0>(n.at(0))) % 12]) {
                    continue;
                }
            }
            //尝试转位和弦
            auto& testBaseChord = it.second;
            auto len = testBaseChord.size();
            if (len > 0) {
                std::vector<int> testChord;  //用于测试的和弦（包括转位）

                for (int beginTone = 0; beginTone < len; ++beginTone) {
                    //构造和弦
                    int delta = testBaseChord[beginTone];  //现在的和弦根音减去delta等于未转位的根音
                    testChord.clear();
                    int lastNote = -999999;
                    //把beginTone位置放开头
                    for (int i = beginTone; i < len; ++i) {
                        int nowNote = testBaseChord[i] - delta;  //减去delta把beginTone位置移动到0
                        while (nowNote <= lastNote) {
                            nowNote += 12;
                        }
                        testChord.push_back(nowNote);
                        lastNote = nowNote;
                    }
                    for (int i = 0; i < beginTone; ++i) {
                        int nowNote = testBaseChord[i] - delta;
                        while (nowNote <= lastNote) {
                            nowNote += 12;
                        }
                        testChord.push_back(nowNote);
                        lastNote = nowNote;
                    }

                    //计算合成代价
                    float weight = calcEditDist(relative, testChord, act) + baseWeight;
                    //std::cout << weight << " " << beginTone << std::endl;
                    if (first || weight < min_chord_weight) {
                        min_chord_weight = weight;
                        min_chord = it.first;
                        min_base = std::get<0>(n.at(0)) - delta;  //因为是转位和弦，需要减去delta得到根音
                        min_chord_inverted = beginTone;
                    }
                    first = false;
                }
            }
        }
    });
    return std::make_tuple(min_base, min_chord, min_chord_inverted, min_chord_weight);
}

template <chord_map_c T>
inline std::tuple<int, std::string, std::vector<int>, int, float> note2Chord(const T& self, const wmelody_t& seq, float act = 0.5) {
    std::map<int, float> count;
    for (auto it : seq) {
        if (std::get<0>(it) > 0) {
            count[std::get<0>(it)] += std::get<1>(it);
        }
    }
    std::vector<std::tuple<int, float>> noteset;
    for (auto it : count) {
        noteset.push_back(std::make_tuple(it.first, (float)it.second));
    }
    auto chord = noteCount2Chord(self, noteset, act);
    std::vector<int> notes;
    return std::make_tuple(std::get<0>(chord), std::get<1>(chord), notes, std::get<2>(chord), std::get<3>(chord));
}

struct musicSection_t {
    float weight;
    std::string chord_name;
    int chord_base;
    int invert;
    wmelody_t melody;
};

template <chord_map_c T>
void initMusicSection(const T& self, musicSection_t& res, float act = 1.0) {
    auto chord = note2Chord(self, res.melody, act);
    res.chord_base = std::get<0>(chord);
    res.chord_name = std::get<1>(chord);
    res.invert = std::get<3>(chord);
    res.weight = std::get<4>(chord);
}

template <chord_map_c T>
musicSection_t buildMusicSection(const T& self, const wmelody_t& melody, float act = 1.0) {
    musicSection_t res;
    res.melody = melody;
    initMusicSection(self, res, act);
    return res;
}

template <chord_map_c T>
musicSection_t merge(const T& self, const musicSection_t& A, const musicSection_t& B, float act = 1.0) {
    musicSection_t res = A;
    for (auto it : B.melody) {
        res.melody.push_back(it);
    }
    initMusicSection(self, res, act);
    return res;
}

template <chord_map_c chord_map_t, typename musicSection_list = std::list<musicSection_t>>
inline void getMusicSection(
    const chord_map_t& chord_map,
    musicSection_list& musicSection,
    float act = 0.5,
    float sectionWeight = 1.0) {
    auto it = musicSection.begin();
    while (it != musicSection.end()) {
        auto next_it = it;
        ++next_it;
        if (next_it != musicSection.end()) {
            //存在下一个，检测合并
            auto& now = *it;
            auto& next = *next_it;
            auto merged = merge(chord_map, now, next, act);
            float nowWeight = now.weight + next.weight + sectionWeight;
#ifdef MGNR_DEBUG
            printf("nowWeight=%f mergeWeight=%f\n", nowWeight, merged.weight);
#endif
            if (nowWeight >= merged.weight) {
                //确认合并
                now = std::move(merged);
                musicSection.erase(next_it);
            } else {
                ++it;
            }
        } else {
            break;
        }
    }
}

template <chord_map_c chord_map_t>
inline std::list<musicSection_t> getMusicSection(
    const chord_map_t& chord_map,
    wmelody_t& melody,
    int minSectionNum,
    float act = 0.5,
    float sectionWeight = 1.0) {
    std::list<musicSection_t> arr;
    wmelody_t tmp;
    for (auto it : melody) {
        if (tmp.size() >= minSectionNum) {
            arr.push_back(std::move(buildMusicSection(chord_map, tmp, act)));
            tmp.clear();
        }
        tmp.push_back(it);
    }
    if (!tmp.empty()) {
        arr.push_back(std::move(buildMusicSection(chord_map, tmp, act)));
    }
    getMusicSection(chord_map, arr, act, sectionWeight);
    return arr;
}

//level与tone
//level为14进制记谱，相当于把简谱的数字乘以2
//tone  c   c#  d   d#  e   f   f#  g   g#  a   a#  b
//tone  0   1   2   3   4   5   6   7   8   9   10  11
//level 0   1   2   3   4   6   7   8   9   10  11  12
//
//level 0   1   2   3   4   5   6   7   8   9   10  11  12  13
//tone  0   1   2   3   4   4   5   6   7   8   9   10  11  11
inline int getToneLevel(int note) {
    if (note < 0) {
        return 0;
    }
    const static int levelList[] = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12};
    return levelList[note % 12] + (note / 12) * 14;
}

inline int getToneLevel(int note, int baseTone) {
    if (note <= 0) {
        return 0;
    }
    return getToneLevel(note - baseTone);
}
inline int getToneLevelDelta(int A, int B, int baseTone) {
    if (A <= 0 || B <= 0) {
        return -65535;
    }
    return getToneLevel(A, baseTone) - getToneLevel(B, baseTone);
}

inline int getToneFromLevelDelta(int A, int B, int baseTone) {
    if (A < 0 || B == -65535) {
        return 0;
    }
    const static int levelList[] = {0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11};
    int BinC = B - baseTone;       //将B变换到C调
    int delta_oct = A / 14;        //音阶
    int delta_pit = A % 14;        //八度
    int B14 = getToneLevel(BinC);  //将B变换到14进制
    int B14_oct = B14 / 14;
    int B14_pit = B14 % 14;
    B14_oct += delta_oct;
    B14_pit += delta_pit;
    B14_oct += B14_pit / 14;
    B14_pit = B14_pit % 14;
    return B14_oct * 12 + levelList[B14_pit] + baseTone;
}

}  // namespace mgnr::melody2chord
namespace chcpy::melody2chord {

using wmelody_t = std::vector<std::tuple<int, float>>;  //加权旋律

inline void getComp(                                                                               //求组合
    const std::vector<std::tuple<int, float>>& notes,                                              //输入已排序的集合{(音符，音符的数量),...}
    const std::function<void(const std::vector<std::tuple<int, float>>&, float weight)>& callback  //回调函数(音符集合,需要改变的数量)
) {
    //求音符总数
    float sum = 0;
    for (auto it : notes) {
        sum += std::get<1>(it);
    }
    //开始求组合
    int len = notes.size();
#pragma omp parallel for
    for (int i = 0; i < len; ++i) {
        std::vector<std::tuple<int, float>> res{};
        float compCount = 0;
        for (int j = i; j < len; ++j) {
            auto it = notes.at(j);
            res.push_back(it);
            compCount += std::get<1>(it);
        }
        callback(res, sum - compCount);
    }
}

inline void notes2RelativeChord(
    const std::vector<std::tuple<int, float>>& notes,
    std::vector<std::tuple<int, float>>& relativeChord) {
    bool first = true;
    int firstNote = std::get<0>(notes.at(0));
    relativeChord.clear();
    for (auto it : notes) {
        relativeChord.push_back(std::make_tuple(std::get<0>(it) - firstNote, std::get<1>(it)));
    }
}

inline int getNumberNoteId(const char* note) {  //此函数性能低下，请勿在初始化以外的地方使用
    static const char* noteMap_1[] = {"1", "1#", "2", "2#", "3", "4", "4#", "5", "5#", "6", "6#", "7"};
    static const char* noteMap_2[] = {"1", "2b", "2", "3b", "3", "4", "5b", "5", "6b", "6", "7b", "7"};
    static const char* noteMap_3[] = {"1", "#1", "2", "#2", "3", "4", "#4", "5", "#5", "6", "#6", "7"};
    static const char* noteMap_4[] = {"1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"};
    for (int i = 0; i < 12; ++i) {
        if (
            strcmp(note, noteMap_1[i]) == 0 ||
            strcmp(note, noteMap_2[i]) == 0 ||
            strcmp(note, noteMap_3[i]) == 0 ||
            strcmp(note, noteMap_4[i]) == 0) {
            return i;
        }
    }
    return 0;
}
inline std::vector<int> getChordArray(const std::vector<const char*>& arr) {
    int last = -1;
    std::vector<int> res;
    for (auto it : arr) {
        int note = getNumberNoteId(it);
        while (note <= last) {
            note += 12;
        }
        last = note;
        res.push_back(note);
    }
    return res;
}

template <typename T>  //要求必须有chord_map成员
concept chord_map_c = requires(T a) {
    a.chord_map;
};

struct chordMap {
    std::map<std::string, std::vector<int>> chord_map;
    inline void addChord(const char* name, const std::vector<const char*>& chord) {
        auto ch = getChordArray(chord);
#ifdef MGNR_DEBUG
        printf("%s\t", name);
        for (auto it : ch) {
            printf("%d\t", it);
        }
        printf("\n");
#endif
        chord_map[name] = ch;
    }
    inline chordMap() {
        //addChord("single",  {"0"});
        addChord("M3", {"1", "3", "5"});
        addChord("m3", {"1", "b3", "5"});
        addChord("-5", {"1", "3", "b5"});
        addChord("ang", {"1", "3", "#5"});
        addChord("dim", {"1", "b3", "b5", "6"});
        addChord("sus", {"1", "4", "5"});
        addChord("6", {"1", "3", "5", "6"});
        addChord("m6", {"1", "b3", "5", "6"});
        addChord("7", {"1", "3", "5", "b7"});
        addChord("maj7", {"1", "3", "5", "7"});
        addChord("m7", {"1", "b3", "5", "b7"});
        addChord("m#7", {"1", "b3", "5", "7"});
        addChord("7+5", {"1", "3", "#5", "b7"});
        addChord("7-5", {"1", "3", "b5", "b7"});
        addChord("m7-5", {"1", "b3", "b5", "b7"});
        addChord("7sus4", {"1", "3", "5", "b7", "4"});
        addChord("7/6", {"1", "3", "5", "b7", "6"});
        addChord("79", {"1", "3", "5", "7", "2"});
        addChord("maj9", {"1", "3", "5", "7", "2"});
        addChord("9", {"1", "3", "5", "b7", "2"});
        addChord("9+5", {"1", "3", "#5", "b7", "2"});
        addChord("9-5", {"1", "3", "b5", "b7", "2"});
        addChord("m9", {"1", "b3", "5", "b7", "2"});
        addChord("7+9", {"1", "3", "5", "b7", "#2"});
        addChord("m9#7", {"1", "b3", "5", "b7", "2"});
        addChord("7b9", {"1", "3", "5", "b7", "b2"});
        addChord("7-9+5", {"1", "3", "#5", "b7", "b2"});
        addChord("7-9-5", {"1", "3", "b5", "b7", "b2"});
        addChord("69", {"1", "3", "5", "6", "2"});
        addChord("m69", {"1", "b3", "5", "6", "2"});
        addChord("11", {"1", "3", "5", "b7", "2", "4"});
        addChord("m11", {"1", "b3", "5", "b7", "2", "4"});
        addChord("11+", {"1", "3", "5", "b7", "2", "#4"});
        addChord("13", {"1", "3", "5", "b7", "2", "4", "6"});
        addChord("13-9", {"1", "3", "5", "b7", "b2", "4", "6"});
        addChord("13-9-5", {"1", "3", "b5", "b7", "b2", "4", "6"});
    }
};

template <chord_map_c T>
bool inChordMap(const T& self, std::vector<int>& chord) {
    bool first = true;
    int start = 0;
    std::vector<int> relativeChord;
    for (auto it : chord) {
        if (first) {
            start = it;
        }
        relativeChord.push_back(it - start);
        first = false;
    }
#ifdef MGNR_DEBUG
    for (auto it : relativeChord) {
        printf("%d ", it);
    }
    printf("\n");
#endif
    for (auto it : self.chord_map) {
        if (it.second == relativeChord) {
            return true;
        }
    }
    return false;
}

template <chord_map_c T>
inline std::tuple<int, std::string, float> note2Chord(const T& self, const std::vector<std::tuple<int, float>>& notes, float act = 1.0) {
    if (notes.empty()) {
        return std::make_tuple(0, "_", 0);
    }
    std::string min_chord;
    float min_base;
    float min_chord_weight;
    std::mutex locker;
    bool first = true;
    getComp(notes, [&](const std::vector<std::tuple<int, float>>& n, float baseWeight) {
        std::vector<std::tuple<int, float>> relative;
        notes2RelativeChord(n, relative);
#ifdef MGNR_DEBUG
        printf("note:");
        for (auto it : relative) {
            printf("%d:%f ", std::get<0>(it), std::get<1>(it));
        }
        printf("\n");
#endif
        for (auto& it : self.chord_map) {
#ifdef MGNR_DEBUG
            printf("chord_map:%s ", it.first.c_str());
            for (auto iit : it.second) {
                printf("%d ", iit);
            }
#endif
            float weight = calcEditDist(relative, it.second, act) + baseWeight;
#ifdef MGNR_DEBUG
            printf("=>%f(%f)\n", weight, baseWeight);
#endif
            locker.lock();
            if (first || weight < min_chord_weight) {
                min_chord_weight = weight;
                min_chord = it.first;
                min_base = std::get<0>(n.at(0));
            }
            first = false;
            locker.unlock();
        }
    });
    return std::make_tuple(min_base, min_chord, min_chord_weight);
}

template <chord_map_c T>
inline std::tuple<int, std::string, std::vector<int>, float> note2Chord(const T& self, const std::vector<int>& seq, float act = 0.5, int times = 1) {
    std::map<int, int> count;
    for (auto it : seq) {
        if (it > 0) {
            count[it] += times;
        }
    }
    std::vector<std::tuple<int, float>> noteset;
    for (auto it : count) {
        noteset.push_back(std::make_tuple(it.first, (float)it.second));
    }
    auto chord = note2Chord(self, noteset, act);
    std::vector<int> notes;
    return std::make_tuple(std::get<0>(chord), std::get<1>(chord), notes, std::get<2>(chord));
}

struct musicSection_t {
    float weight = 0;
    std::string chord_name{};
    int chord_base = 0;
    std::vector<int> melody{};
};

template <chord_map_c T>
void initMusicSection(const T& self, musicSection_t& res, float act = 1.0, int times = 1) {
    auto chord = note2Chord(self, res.melody, act, times);
    res.chord_base = std::get<0>(chord);
    res.chord_name = std::get<1>(chord);
    res.weight = std::get<3>(chord);
}

template <chord_map_c T>
musicSection_t buildMusicSection(const T& self, const std::vector<int>& melody, float act = 1.0, int times = 1) {
    musicSection_t res;
    res.melody = melody;
    initMusicSection(self, res, act, times);
    return res;
}

template <chord_map_c T>
musicSection_t merge(const T& self, const musicSection_t& A, const musicSection_t& B, float act = 1.0, int times = 1) {
    musicSection_t res = A;
    for (auto it : B.melody) {
        res.melody.push_back(it);
    }
    initMusicSection(self, res, act, times);
    return res;
}

template <typename musicSection_list>
concept musicSection_list_c = requires(musicSection_list l) {
    l.begin()->weight;
    l.begin()->chord_name.c_str();
    l.begin()->chord_base;
    l.begin()->melody.begin();
};

template <chord_map_c chord_map_t, musicSection_list_c musicSection_list = std::list<musicSection_t>>
inline void getMusicSection(
    const chord_map_t& chord_map,
    musicSection_list& musicSection,
    float act = 0.5,
    float sectionWeight = 1.0,
    int times = 1) {
    auto it = musicSection.begin();
    while (it != musicSection.end()) {
        auto next_it = it;
        ++next_it;
        if (next_it != musicSection.end()) {
            //存在下一个，检测合并
            auto& now = *it;
            auto& next = *next_it;
            auto merged = merge(chord_map, now, next, act, times);
            float nowWeight = now.weight + next.weight + sectionWeight;
#ifdef MGNR_DEBUG
            printf("nowWeight=%f mergeWeight=%f\n", nowWeight, merged.weight);
#endif
            if (nowWeight >= merged.weight) {
                //确认合并
                now = std::move(merged);
                musicSection.erase(next_it);
            } else {
                ++it;
            }
        } else {
            break;
        }
    }
}

template <chord_map_c chord_map_t>
inline std::list<musicSection_t> getMusicSection(
    const chord_map_t& chord_map,
    const std::vector<int>& melody,
    int minSectionNum,
    float act = 0.5,
    float sectionWeight = 1.0,
    int times = 1) {
    std::list<musicSection_t> arr;
    std::vector<int> tmp;
    for (auto it : melody) {
        if (tmp.size() >= minSectionNum) {
            arr.push_back(std::move(buildMusicSection(chord_map, tmp, act, times)));
            tmp.clear();
        }
        tmp.push_back(it);
    }
    if (!tmp.empty()) {
        arr.push_back(std::move(buildMusicSection(chord_map, tmp, act, times)));
    }
    getMusicSection(chord_map, arr, act, sectionWeight);
    return arr;
}

//level与tone
//level为14进制记谱，相当于把简谱的数字乘以2
//tone  c   c#  d   d#  e   f   f#  g   g#  a   a#  b
//tone  0   1   2   3   4   5   6   7   8   9   10  11
//level 0   1   2   3   4   6   7   8   9   10  11  12
//
//level 0   1   2   3   4   5   6   7   8   9   10  11  12  13
//tone  0   1   2   3   4   4   5   6   7   8   9   10  11  11
inline int getToneLevel(int note) {
    if (note < 0) {
        return 0;
    }
    const static int levelList[] = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12};
    return levelList[note % 12] + (note / 12) * 14;
}

inline int getToneLevel(int note, int baseTone) {
    if (note <= 0) {
        return 0;
    }
    return getToneLevel(note - baseTone);
}
inline int getToneLevelDelta(int A, int B, int baseTone) {
    if (A <= 0 || B <= 0) {
        return -65535;
    }
    return getToneLevel(A, baseTone) - getToneLevel(B, baseTone);
}

inline int getToneFromLevelDelta(int A, int B, int baseTone) {
    if (A < 0 || B == -65535) {
        return 0;
    }
    const static int levelList[] = {0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11};
    int BinC = B - baseTone;       //将B变换到C调
    int delta_oct = A / 14;        //音阶
    int delta_pit = A % 14;        //八度
    int B14 = getToneLevel(BinC);  //将B变换到14进制
    int B14_oct = B14 / 14;
    int B14_pit = B14 % 14;
    B14_oct += delta_oct;
    B14_pit += delta_pit;
    B14_oct += B14_pit / 14;
    B14_pit = B14_pit % 14;
    return B14_oct * 12 + levelList[B14_pit] + baseTone;
}

}  // namespace chcpy::melody2chord
