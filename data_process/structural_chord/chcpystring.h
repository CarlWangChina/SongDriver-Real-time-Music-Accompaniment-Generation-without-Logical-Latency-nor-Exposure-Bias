#pragma once
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace chcpy {

class string;
using stringlist = std::vector<string>;

inline std::string& replace_all(std::string& str, const std::string& old_value, const std::string& new_value) {
    while (true) {
        std::string::size_type pos(0);
        if ((pos = str.find(old_value)) != std::string::npos) {
            str.replace(pos, old_value.length(), new_value);
        } else {
            break;
        }
    }
    return str;
}

class string : public std::string {
   public:
    inline string() {}

    inline string(const string& s)
        : std::string(s) {}

    inline string(const std::string& s)
        : std::string(s) {}

    inline string(const char* s)
        : std::string(s) {}

    inline stringlist split(const std::string& seprate) const {
        const auto& s = *this;
        stringlist ret;
        int seprate_len = seprate.length();
        int start = 0;
        int index;
        while ((index = s.find(seprate, start)) != -1) {
            ret.push_back(s.substr(start, index - start));
            start = index + seprate_len;
        }
        if (start < s.length())
            ret.push_back(s.substr(start, s.length() - start));
        return ret;
    }
    inline stringlist split(const char delim) const {
        stringlist elems;
        std::stringstream ss(*this);
        string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }
    inline string replace(const std::string& o, const std::string& n) const {
        string res = *this;
        replace_all(res, o, n);
        return res;
    }
    inline string simplified() const {
        string res;
        for (auto c : *this) {
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\0') {
                res += c;
            }
        }
        return res;
    }
    inline string trimmed() const {
        string s = *this;
        if (s.empty()) {
            return s;
        }
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
        return s;
    }
    inline string mid(int begin, int num) const {
        string res;
        int index = 0;
        int count = 0;
        for (auto c : *this) {
            if (index >= begin) {
                res += c;
                ++count;
                if (count >= num) {
                    break;
                }
            }
            ++index;
        }
        return res;
    }
    inline int toInt() const {
        int res = 0;
        std::istringstream ss(*this);
        ss >> res;
        return res;
    }
    inline float toFloat() const {
        float res = 0;
        std::istringstream ss(*this);
        ss >> res;
        return res;
    }
    static inline string number(int i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", i);
        return string(buf);
    }
};

template <class S, class T>
inline std::string join(std::vector<T>& elems, S& delim) {
    if (elems.empty()) {
        return std::string();
    }
    std::stringstream ss;
    typename std::vector<T>::iterator e = elems.begin();
    ss << *e++;
    for (; e != elems.end(); ++e) {
        ss << delim << *e;
    }
    return ss.str();
}

}  // namespace chcpy