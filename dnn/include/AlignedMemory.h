#pragma once
#include <cstdlib>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "dnnl.hpp"
#include "dnnl_debug.h"

#include "fastmem.h"
#include "ParallelFor.h"

namespace dnn
{
    template <typename T> class AlignedMemory
	{
		typedef typename std::size_t size_type;

	protected:
		std::unique_ptr<dnnl::memory> arrPtr = nullptr;
		T* dataPtr = nullptr;
		size_type nelems = 0;
		dnnl::memory::desc description;

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
		AlignedMemory() NOEXCEPT { }
		AlignedMemory(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T(0)) NOEXCEPT
		{
			if (md)
			{
				AlignedMemory::release();

				arrPtr = std::make_unique<dnnl::memory>(md, engine);
				if (arrPtr)
				{
					dataPtr = static_cast<T*>(arrPtr->get_data_handle());
					nelems = md.get_size() / sizeof(T);

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
		inline auto memory() noexcept { return arrPtr.get(); }
		inline auto data() noexcept { return dataPtr; }
		inline auto data() const noexcept { return dataPtr; }
		inline auto size() const noexcept { return nelems; }
		auto desc() { return description; }
		void resizeMem(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T(0)) NOEXCEPT
		{
			if (md)
			{
				if (md.get_size() / sizeof(T) == nelems)
					return;

				AlignedMemory::release();

				if (md.get_size() / sizeof(T) > 0)
				{
					arrPtr = std::make_unique<dnnl::memory>(md, engine);
					if (arrPtr)
					{
						dataPtr = static_cast<T*>(arrPtr->get_data_handle());
						nelems = md.get_size() / sizeof(T);
						
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
						
						description = md;
					}
				}
			}
		}
		void resize(const size_type n, const size_type c, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type d, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(d), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		inline T& operator[] (size_type i) NOEXCEPT { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const NOEXCEPT { return dataPtr[i]; }
		inline auto empty() const noexcept { return nelems == 0; }
	};
}
