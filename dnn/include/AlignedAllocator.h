#pragma once

#include <cstdlib>
#include <new>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cstddef>

#ifdef __MINGW32__
#include <mm_malloc.h>
#endif

#if defined(_MSC_VER)
    #define DNN_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #if __has_attribute(always_inline)
        #define DNN_INLINE inline __attribute__((always_inline))
    #else
        #define DNN_INLINE inline
    #endif
#else
    #define DNN_INLINE inline
#endif

namespace dnn
{
    /**
     * @brief An STL-compatible allocator that ensures memory is aligned to a specific boundary.
     * @tparam T The type of object to allocate.
     * @tparam Alignment The byte alignment (must be a power of two).
     */
    template <typename T, std::size_t Alignment>
    class AlignedAllocator
    {
        static_assert(__cplusplus >= 201703L, "AlignedAllocator requires C++17 or later");
        static_assert(sizeof(T) > 0, "Type T must have non-zero size");
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
        static_assert(Alignment >= sizeof(void*), "Alignment cannot be less than pointer size");

    public:
        using value_type      = T;
        using pointer         = T*;
        using const_pointer   = const T*;
        using size_type       = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference       = T&;
        using const_reference = const T&;
        using is_always_equal = std::true_type;

        // Rebind is still useful for older compilers/specific STL implementations
        template <typename U>
        struct rebind { 
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() noexcept = default;
        
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        ~AlignedAllocator() = default;

        [[nodiscard]] DNN_INLINE pointer allocate(std::size_t n)
        {
#ifdef DEBUG
            assert(n > 0 && "Cannot allocate zero size");
#endif
            if (n == 0) return nullptr;

            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::bad_array_new_length();
            }

            const std::size_t total_size = n * sizeof(T);
            
            void* p = AlignedAlloc(Alignment, total_size);

            if (!p) {
                throw std::bad_alloc();
            }

            return static_cast<pointer>(p);
        }

        DNN_INLINE void deallocate(pointer p, std::size_t /*n*/) noexcept
        {
            if (p) {
                AlignedFree(p);
            }
        }

        [[nodiscard]] static constexpr size_type max_size() noexcept
        { 
            return std::numeric_limits<std::size_t>::max() / sizeof(T); 
        }

    protected:
        static constexpr std::size_t RoundUp(std::size_t size, std::size_t align) noexcept
        {
            return (size + align - 1) & ~(align - 1);
        }

        DNN_INLINE void* AlignedAlloc(std::size_t align, std::size_t size) const
        {
            const std::size_t adjusted_size = RoundUp(size, align);
#ifdef DEBUG
            assert((adjusted_size % align == 0) && "Size must be multiple of alignment");
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
            return ::_aligned_malloc(adjusted_size, align);
#elif defined(__ANDROID__)
            return ::memalign(align, adjusted_size);
#elif defined(__MINGW32__)
            return ::_mm_malloc(adjusted_size, align);
#else // POSIX
            void* p = nullptr;
            p = ::aligned_alloc(align, adjusted_size);
            if (!p) {
                ::posix_memalign(&p, align, adjusted_size);
            }
            return p;
#endif
        }

        DNN_INLINE void AlignedFree(pointer ptr) const noexcept
        {
#if defined(_WIN32) || defined(__CYGWIN__)
            ::_aligned_free(ptr);
#elif defined(__MINGW32__)
            ::_mm_free(ptr);
#else
            ::free(ptr);
#endif
        }
    };

    template <typename T, typename U, std::size_t A>
    bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

    template <typename T, typename U, std::size_t A>
    bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }
}
