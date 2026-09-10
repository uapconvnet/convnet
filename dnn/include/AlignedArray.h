
#include <cstdlib>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "fastmem.h"

namespace dnn
{

	struct aligned_free
	{
		aligned_free() = default;

		void operator()(void* ptr)
		{
			if (ptr)
			{
#if defined(_WIN32) || defined(__CYGWIN__)
				::_aligned_free(ptr);
#elif defined(__MINGW32__)
				::_mm_free(ptr);
#else
				::free(ptr);
#endif
			}
		}
	};
	
	template<typename T>
	T* aligned_malloc(std::size_t size, std::size_t alignment) 
	{ 
#if defined(_WIN32) || defined(__CYGWIN__)
		return static_cast<T*>(::_aligned_malloc(size * sizeof(T), alignment));
#elif defined(__ANDROID__)
		return static_cast<T*>(::memalign(size * sizeof(T), alignment));
#elif defined(__MINGW32__)
		return  static_cast<T*>(::_mm_malloc(size * sizeof(T), alignment));
#else  // posix assumed
		return static_cast<T*>(::aligned_alloc(alignment, size * sizeof(T)));
#endif
	}

	template<class T> using unique_ptr_aligned = std::unique_ptr<T, aligned_free>;

	template<class T, std::size_t alignment> 
	unique_ptr_aligned<T> aligned_unique_ptr(std::size_t size, std::size_t align) { return unique_ptr_aligned<T>(static_cast<T*>(aligned_malloc<T>(size, align))); }

	template <typename T, std::size_t alignment> class AlignedArray
	{
		typedef typename std::size_t size_type;

	protected:
		unique_ptr_aligned<T> arrPtr = nullptr;
		T* dataPtr = nullptr;
		size_type nelems = 0;

	public:
	    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, unsigned char>::value || std::is_same<T, char>::value || std::is_same<T, unsigned int>::value || std::is_same<T, int>::value || std::is_same<T, unsigned long>::value || std::is_same<T, long>::value, "T has unsupported type");
		void release() NOEXCEPT
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;
		}
		AlignedArray() NOEXCEPT	{ }
		AlignedArray(const size_type elements, const T value = T(0)) NOEXCEPT
		{
			AlignedArray::release();

			arrPtr = aligned_unique_ptr<T, alignment>(elements, alignment);
			if (arrPtr)
			{
				dataPtr = arrPtr.get();
				nelems = elements;

				if (value == T(0))
					fast_memzero(dataPtr, nelems * sizeof(T));
				else
				{
					if constexpr (std::is_same<T, unsigned char>::value)
						fast_memset(dataPtr, value, nelems * sizeof(T));
					else if constexpr (std::is_same<T, char>::value)
						fast_memset(dataPtr, static_cast<unsigned char>(value), nelems * sizeof(T));
					else if constexpr (std::is_same<T, unsigned int>::value)
						fast_memset_4B(dataPtr, value, nelems);
					else if constexpr (std::is_same<T, int>::value)
						fast_memset_4B(dataPtr, static_cast<unsigned int>(value), nelems);
					else
						PRAGMA_OMP_SIMD()
						for (auto i = 0ull; i < nelems; i++)
							dataPtr[i] = value;
				}
			}
		}
		inline auto data() noexcept { return dataPtr; }
		inline auto data() const noexcept { return dataPtr; }
		inline auto size() const noexcept { return nelems; }
		void resize(size_type elements, const T value = T(0)) NOEXCEPT
		{ 
			if (elements == nelems)
				return;

			AlignedArray::release();
			
			if (elements > 0)
			{
				arrPtr = aligned_unique_ptr<T, alignment>(elements, alignment);
				if (arrPtr)
				{
					dataPtr = arrPtr.get();
					nelems = elements;
					
					if (value == T(0))
						fast_memzero(dataPtr, nelems * sizeof(T));
					else
					{
						if constexpr (std::is_same<T, unsigned char>::value)
							fast_memset(dataPtr, value, nelems * sizeof(T));
						else if constexpr (std::is_same<T, char>::value)
							fast_memset(dataPtr, static_cast<unsigned char>(value), nelems * sizeof(T));
						else if constexpr (std::is_same<T, unsigned int>::value)
							fast_memset_4B(dataPtr, value, nelems);
						else if constexpr (std::is_same<T, int>::value)
							fast_memset_4B(dataPtr, static_cast<unsigned int>(value), nelems);
						else
							PRAGMA_OMP_SIMD()
							for (auto i = 0ull; i < nelems; i++)
								dataPtr[i] = value;
					}
				}
			}		
		}
		inline T& operator[] (size_type i) NOEXCEPT { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const NOEXCEPT { return dataPtr[i]; }
		inline auto empty() const noexcept { return nelems == 0; }
	};
}
