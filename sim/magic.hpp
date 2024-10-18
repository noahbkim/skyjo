#pragma once

#include <array>
#include <tuple>
#include <type_traits>

struct enumerate {};
namespace std {
    template<size_t I>
    constexpr size_t get(const enumerate& e) {
        return I;
    }
}

template<typename T>
struct starmap_meta { static_assert(false); };

template<typename First, typename... Rest>
struct starmap_meta<std::tuple<First, Rest...>> {
    static constexpr size_t size = 1 + sizeof...(Rest);
    using element = First;
};

template<typename T, size_t Size>
struct starmap_meta<std::array<T, Size>> {
    static constexpr size_t size = Size;
    using element = T;
};

template<>
struct starmap_meta<enumerate> {
    using element = size_t;
};

template<size_t I, typename F, typename Iterable, typename... Iterables>
decltype(auto) star(F&& f, Iterable&& iterable, Iterables&&... iterables) {
    return f(std::get<I>(iterable), std::get<I>(iterables)...);
}

template<typename F, typename Iterable, typename... Iterables>
decltype(auto) starmap(F&& f, Iterable&& iterable, Iterables&&... iterables) {
    constexpr size_t size = starmap_meta<std::decay_t<Iterable>>::size;
    return [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
        using R = decltype(f(
            std::declval<typename starmap_meta<std::decay_t<Iterable>>::element&>(),
            std::declval<typename starmap_meta<std::decay_t<Iterables>>::element&>()...
        ));
        if constexpr (std::is_same_v<R, void>) {
            (star<Indices>(
                std::forward<F>(f),
                std::forward<Iterable>(iterable),
                std::forward<Iterables>(iterables)...
            ), ...);
        } else {
            return std::array<R, size>{{
                star<Indices>(
                    std::forward<F>(f),
                    std::forward<Iterable>(iterable),
                    std::forward<Iterables>(iterables)...
                )...
            }};
        }
    }(std::make_index_sequence<size>{});
}

template<typename Index = size_t, typename Iterator, typename F>
Index imax(Iterator begin, Iterator end, F key = [](const auto& value) { return value; }, Index empty = {}) {
    if (begin == end) {
        return empty;
    } else {
        Index max_index = 0;
        auto max_value = key(*begin);
        Iterator cursor = begin;
        while (++cursor != end) {
            auto new_value = key(*cursor);
            if (new_value > max_value) {
                max_value = new_value;
                max_index = cursor - begin;
            }
        }
        return max_index;
    }
}

template<typename Index = size_t, typename Iterator, typename F>
Index imin(Iterator begin, Iterator end, F key = [](const auto& value) { return value; }, Index empty = {}) {
    if (begin == end) {
        return empty;
    } else {
        Index min_index = 0;
        auto min_value = key(*begin);
        Iterator cursor = begin;
        while (++cursor != end) {
            auto new_value = key(*cursor);
            if (new_value < min_value) {
                min_value = new_value;
                min_index = cursor - begin;
            }
        }
        return min_index;
    }
}

template<typename Base, typename Tuple>
decltype(auto) upcast(Tuple&& tuple) {
    constexpr size_t size = std::tuple_size_v<std::decay_t<Tuple>>;
    return [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
        return std::array<Base*, size>{{dynamic_cast<Base*>(&std::get<Indices>(tuple))...}};
    }(std::make_index_sequence<size>{});
}

template<typename Base, typename... Ts>
struct vtuple : public std::array<Base*, sizeof...(Ts)> {
    static constexpr size_t size = sizeof...(Ts);

    std::tuple<Ts...> inner;

    template<typename... Args>
    explicit vtuple(Args&&... args)
        : std::array<Base*, sizeof...(Ts)>{upcast<Base>(this->inner)}
        , inner{std::forward<Args>(args)...} {}
};
