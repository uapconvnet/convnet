#pragma once

#include <cstdlib>
#include <new>       // Required for placement new
#include <limits>    // For std::numeric_limits
#include <stdexcept>
#include <type_traits>
#include <cstddef>

#ifdef __MINGW32__
#include <mm_malloc.h>
#endif

// Improved Macro: Use constexpr/inline where possible instead of heavy macros
#if defined(_MSC_VER)
    #define DNN_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #define DNN_INLINE inline __attribute__((always_inline))
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
        // Static assertion to ensure alignment is a power of two at compile time
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two.");

    public:
        using value_type      = T;
        using pointer         = T*;
        using const_pointer   = const T*;
        using size_type       = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference       = T&;
        using const_reference = const T&;

        // Rebind is still useful for older compilers/specific STL implementations
        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        // Constructors
        AlignedAllocator() noexcept = default;
        
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        ~AlignedAllocator() = default;

        // --- Core Allocation Logic ---

        [[nodiscard]] DNN_INLINE pointer allocate(std::size_t n)
        {
            if (n == 0) return nullptr;

            // 1. Overflow Check: Ensure size * sizeof(T) doesn't wrap around
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::bad_array_new_length();
            }

            const std::size_t total_size = n * sizeof(T);
            
            // 2. Perform aligned allocation
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

        // --- Utilities ---

        [[nodiscard]] DNN_INLINE size_type max_size() const noexcept 
        { 
            return std::numeric_limits<std::size_t>::max() / sizeof(T); 
        }

        DNN_INLINE pointer address(reference value) const noexcept { return std::addressof(value); }
        DNN_INLINE const_pointer address(const_reference value) const noexcept { return std::addressof(value); }

    protected:
        // Helper to ensure the requested size is a multiple of the alignment (required by POSIX)
        static constexpr std::size_t RoundUp(std::size_t size, std::size_t align) noexcept
        {
            return (size + align - 1) & ~(align - 1);
        }

        DNN_INLINE void* AlignedAlloc(std::size_t align, std::size_t size) const
        {
            const std::size_t adjusted_size = RoundUp(size, align);

#if defined(_WIN32) || defined(__CYGWIN__)
            return ::_aligned_malloc(adjusted_size, align);
#elif defined(__ANDROID__)
            return ::memalign(align, adjusted_size);
#elif defined(__MINGW32__)
            return ::_mm_malloc(adjusted_size, align);
#else // POSIX
            // C++17 aligned_alloc requires size to be a multiple of alignment
            return ::aligned_alloc(align, adjusted_size);
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

    // Equality operators are simplified (stateless allocator)
    template <typename T, typename U, std::size_t A>
    bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

    template <typename T, typename U, std::size_t A>
    bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }
}