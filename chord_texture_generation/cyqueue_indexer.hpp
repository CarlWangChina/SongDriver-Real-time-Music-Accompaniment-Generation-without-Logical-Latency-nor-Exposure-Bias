#ifndef AUTOCHORD_CYQUEUE_IDEXER
#define AUTOCHORD_CYQUEUE_IDEXER
#include <functional>
namespace autochord{

struct cyqueue_indexer_t{
    int index_len;
    int index;
};

template<typename T>
void cyqueue_indexer_init(T & self,int len) {
    self.index_len = len;
    self.index = 0;
}

template<typename T>
void cyqueue_indexer_foreach (T & self,std::function<bool(int)> callback) {
    for (int i = self.index - 1; i >= 0; --i) {
        if (callback(i))
            return;
    }
    for (int i = self.index_len - 1; i >= self.index; --i) {
        if (callback(i))
            return;
    }
}

template<typename T>
void cyqueue_indexer_foreach_r(T & self,std::function<bool(int)> callback) {
    for (int i = self.index - 1; i >= 0; --i) {
        if (callback(i))
            return;
    }
    for (int i = self.index_len - 1; i >= self.index; --i) {
        if (callback(i))
            return;
    }
}

template<typename T>
void cyqueue_indexer_next(T & self) {
    ++self.index;
    if (self.index >= self.index_len) {
        self.index = 0;
    }
}

template<typename T>
int cyqueue_indexer_last(T & self,int id) {
    id = id % self.index_len;
    int r = self.index - id;
    if (r < 0) {
        r += self.index_len;
    }
    return r;
}

template<typename T>
int cyqueue_indexer_at(T & self,int id) {
    return (id+self.index) % self.index_len;
}

}
#endif