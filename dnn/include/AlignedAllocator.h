#pragma once
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <utility>
#include <stdexcept>

#ifdef __MINGW32__
#include <mm_malloc.h>
#endif

#if defined(__VEC__)
#define DNN_INLINE inline
#elif defined __has_attribute
#if __has_attribute(always_inline)
#define DNN_INLINE inline __attribute__((always_inline))
#else
#define DNN_INLINE inline
#endif
#elif defined(_MSC_VER)
#define DNN_INLINE inline __forceinline
#else
#define DNN_INLINE inline
#endif

namespace dnn
{
	template <typename T, std::size_t alignment>
	class AlignedAllocator
	{
	public:
		typedef T value_type;
		typedef T* pointer;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef const T* const_pointer;

		template <typename U>
		struct rebind
		{
			typedef AlignedAllocator<U, alignment> other;
		};

		DNN_INLINE AlignedAllocator() noexcept {};

		template <typename U>
		DNN_INLINE AlignedAllocator(const AlignedAllocator<U, alignment>&) noexcept {};

		DNN_INLINE ~AlignedAllocator() {};

		DNN_INLINE const_pointer address(const_reference value) const noexcept { return std::addressof(value); }

		DNN_INLINE pointer address(reference value) const noexcept { return std::addressof(value); }

		DNN_INLINE pointer allocate(const size_type size, const void* = nullptr)
		{
			void* p = AlignedAlloc(alignment, sizeof(T) * size);

			if (!p && size > 0ull)
				throw std::runtime_error("failed to allocate");

			return static_cast<pointer>(p);
		}

		DNN_INLINE size_type max_size() const noexcept { return ~static_cast<std::size_t>(0ull) / sizeof(T); }

		DNN_INLINE void deallocate(pointer ptr, size_type) { AlignedFree(ptr); }

		template <class U, class V>
		DNN_INLINE void construct(U* ptr, const V& value)
		{
			void* p = ptr;
			::new (p) U(value);
		}

		template <class U, class... Args>
		DNN_INLINE void construct(U* ptr, Args &&... args)
		{
			void* p = ptr;
			::new (p) U(std::forward<Args>(args)...);
		}

		template <class U>
		DNN_INLINE void construct(U* ptr)
		{
			void* p = ptr;
			::new (p) U();
		}

		template <class U>
		DNN_INLINE void destroy(U* ptr)
		{
			ptr->~U();
		}

	protected:
		DNN_INLINE size_type DIVALIGN([[maybe_unused]] const size_type align, const size_type size) const noexcept
		{
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
			return size;
#else
			if (size % align == 0ull)
				return size;

			return ((size / align) + 1ull) * align;
#endif
		}

		DNN_INLINE void* AlignedAlloc(const size_type align, const size_type size) const
		{
#if defined(_WIN32) || defined(__CYGWIN__)
			return ::_aligned_malloc(DIVALIGN(align, size), align);
#elif defined(__ANDROID__)
			return ::memalign(align, DIVALIGN(align, size));
#elif defined(__MINGW32__)
			return ::_mm_malloc(DIVALIGN(align, size), align);
#else  // posix assumed
			return ::aligned_alloc(align, DIVALIGN(align, size));
#endif
		}

		DNN_INLINE void AlignedFree(pointer ptr)
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

	template <typename T1, typename T2, std::size_t alignment>
	DNN_INLINE bool operator==(const AlignedAllocator<T1,alignment>&, const AlignedAllocator<T2,alignment>&) noexcept { return true; }

	template <typename T1, typename T2, std::size_t alignment>
	DNN_INLINE bool operator!=(const AlignedAllocator<T1,alignment>&, const AlignedAllocator<T2,alignment>&) noexcept { return false; }
}