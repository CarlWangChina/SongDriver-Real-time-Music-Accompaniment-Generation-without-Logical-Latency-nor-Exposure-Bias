#pragma once
#include <coroutine>
#include <exception>

namespace midiSearch {

//封装generator
template <typename T>
class generator {
   public:
    struct promise_type {
        generator get_return_object() {
            return generator{handle_type::from_promise(*this)};
        }

        static generator get_return_object_on_allocation_failure() {
            return generator{nullptr};
        }

        auto initial_suspend() {
            return std::suspend_always{};
        }

        auto final_suspend() noexcept {
            return std::suspend_always{};
        }

        auto yield_value(T value) {
            current_value = value;

            return std::suspend_always{};
        }

        void return_void() {}

        void unhandled_exception() {
            std::terminate();
        }

        T current_value;
    };
    using handle_type = std::coroutine_handle<promise_type>;

    struct iterator_end {};

    struct iterator {
        iterator(handle_type handle_)
            : handle(handle_) {}

        void operator++() {
            handle.resume();
        }

        T operator*() {
            return std::move(handle.promise().current_value);
        }

        bool operator==(iterator_end) {
            return handle.done();
        }

        bool operator!=(iterator_end) {
            return !handle.done();
        }

        handle_type handle;
    };

    generator(handle_type handle_)
        : handle(handle_) {}
    generator(const generator& other) = delete;
    generator(generator&& other) noexcept
        : handle(other.handle) {
        other.handle = nullptr;
    }
    ~generator() {
        if (handle) {
            handle.destroy();
        }
    }

    bool has_next() {
        if (handle) {
            handle.resume();

            return !handle.done();
        }

        return false;
    }

    T get_value() {
        return std::move(handle.promise().current_value);
    }

    iterator begin() {
        handle.resume();
        return iterator{handle};
    }

    iterator_end end() {
        return iterator_end{};
    }

   private:
    handle_type handle;
};

}  // namespace midiSearch