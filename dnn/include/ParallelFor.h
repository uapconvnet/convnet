#pragma once
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
#include <assert.h>
#include <omp.h>
#include "fastmem.h"
#else
#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <future>
#include <thread>
#endif

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CHAIn2(a, b) a b
#define CHAIN2(a, b) CHAIn2(a, b)

#ifdef _MSC_VER
#define PRAGMA_MACRo(x) __pragma(x)
#else
#define PRAGMA_MACRo(x) _Pragma(#x)
#endif
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n) PRAGMA_MACRO(omp parallel for collapse(n))
#define PRAGMA_OMP_PARALLEL_THREADS(n) PRAGMA_MACRO(omp parallel num_threads(n))
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n) PRAGMA_MACRO(omp for schedule(static,n))
#define PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(n) PRAGMA_MACRO(omp for schedule(dynamic,n))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define PRAGMA_OMP(...)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)
#define PRAGMA_OMP_PARALLEL_THREADS(n)
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n)
#define PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(n)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

// Disabling OMP SIMD feature for MSVC as it only supports OpenMP 2.0
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)


#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

#ifndef DNN_UNREF_PAR
  #ifdef _MSC_VER
	#define DNN_UNREF_PAR(P) (P)
  #else
	#define DNN_UNREF_PAR(P) (void)P
  #endif
#endif

namespace dnn
{
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP
	struct blocked_range
	{
		typedef size_t const_iterator;

		blocked_range(const size_t begin, const size_t end) : begin_(begin), end_(end) {}
		blocked_range(const int& begin, const int& end) : begin_(begin), end_(end) {}

		const_iterator begin() const { return begin_; }
		const_iterator end() const { return end_; }

	private:
		const size_t begin_;
		const size_t end_;
	};

	template <typename Func>
	void xparallel_for(const size_t begin, const size_t end, const Func& f)
	{
		blocked_range r(begin, end);
		f(r);
	}

	template <typename Func>
	void parallel_for(const size_t begin, const size_t end, const Func& f)
	{
		assert(end >= begin);

		const auto nthreads = std::thread::hardware_concurrency();
		auto blockSize = (end - begin) / nthreads;
		if (blockSize * nthreads < end - begin)
			blockSize++;

		std::vector<std::future<void>> futures;

		auto blockBegin = begin;
		auto blockEnd = blockBegin + blockSize;
		if (blockEnd > end)
			blockEnd = end;

		for (auto i = 0ull; i < nthreads; i++)
		{
			futures.push_back(std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] {	f(blocked_range(blockBegin, blockEnd));	})));

			blockBegin += blockSize;
			if (blockBegin >= end)
				break;

			blockEnd = blockBegin + blockSize;
			if (blockEnd > end)
				blockEnd = end;
		}

		for (auto& future : futures)
			future.wait();
	}

	template <typename T, typename U>
	bool value_representation(U const& value) { return static_cast<U>(static_cast<T>(value)) == value; }

	template <typename Func>
	inline void for_(const size_t begin, const size_t end, const Func& f)
	{
		value_representation<size_t>(end) ? parallel_for(begin, end, f) : xparallel_for(begin, end, f);
	}
#endif

	/* template <typename Func>
	inline void for_i(const size_t range, const Func& f)
	{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
		const auto thrds = static_cast<int>(std::min(range, static_cast<size_t>(omp_get_max_threads())));
		const auto chunk = static_cast<int>(std::ceil(static_cast<double>(range) / static_cast<double>(thrds)));

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
		PRAGMA_OMP_PARALLEL_THREADS(thrds)
		{
			PRAGMA_OMP_FOR_SCHEDULE_STATIC(chunk)
				for (auto i = 0ll; i < static_cast<long long>(range); i++)
					f(i);
		}
#else
#pragma omp parallel for schedule(static,chunk) num_threads(thrds)
		for (auto i = 0ull; i < range; i++)
			f(i);
#endif
#else
		for_(0ull, range, [&](const blocked_range& r)
			{
				for (auto i = r.begin(); i < r.end(); i++)
					f(i);
			});
#endif
	}

	template <typename Func>
	inline void for_i(const size_t range, const size_t threads, const Func& f)
	{
		if (std::min(range, threads) > 1)
		{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
			const auto thrds = static_cast<int>(std::min(range, threads));
			const auto chunk = static_cast<int>(std::ceil(static_cast<double>(range) / static_cast<double>(thrds)));

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
			PRAGMA_OMP_PARALLEL_THREADS(thrds)
			{
				PRAGMA_OMP_FOR_SCHEDULE_STATIC(chunk)
					for (auto i = 0ll; i < static_cast<long long>(range); i++)
						f(i);
			}
#else

#pragma omp parallel for schedule(static,chunk) num_threads(thrds)
			for (auto i = 0ull; i < range; i++)
				f(i);
#endif
#else
			for_(0ull, range, [&](const blocked_range& r)
				{
					for (auto i = r.begin(); i < r.end(); i++)
						f(i);
				});
#endif
		}
		else
			for (auto i = 0ull; i < range; i++)
				f(i);
	} */

	template <typename Func>
	inline void for_i_dynamic(const size_t range, const Func& f)
	{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
		const auto thrds = static_cast<int>(std::min(range, static_cast<size_t>(omp_get_max_threads())));

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
		PRAGMA_OMP_PARALLEL_THREADS(thrds)
		{
			PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(1)
				for (auto i = 0ll; i < static_cast<long long>(range); i++)
					f(i);
		}
#else
#pragma omp parallel for schedule(dynamic,1) num_threads(thrds)
		for (auto i = 0ull; i < range; i++)
			f(i);
#endif
#else
		for_(0ull, range, [&](const blocked_range& r)
			{
				for (auto i = r.begin(); i < r.end(); i++)
					f(i);
			});
#endif
	}

	template <typename Func>
	inline void for_i_dynamic(const size_t range, const size_t threads, const Func& f)
	{
		if (std::min(range, threads) > 1)
		{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
			const auto thrds = static_cast<int>(std::min(range, static_cast<size_t>(omp_get_max_threads())));

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
			PRAGMA_OMP_PARALLEL_THREADS(thrds)
			{
				PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(1)
					for (auto i = 0ll; i < static_cast<long long>(range); i++)
						f(i);
			}
#else
#pragma omp parallel for schedule(dynamic,1) num_threads(thrds)
			for (auto i = 0ull; i < range; i++)
				f(i);
#endif
#else
			DNN_UNREF_PAR(threads);
			for_(0ull, range, [&](const blocked_range& r)
				{
					for (auto i = r.begin(); i < r.end(); i++)
						f(i);
				});
#endif
		}
		else
			for (auto i = 0ull; i < range; i++)
				f(i);
	}

	/* SFINAE helper -- analogue to std::enable_if */
	template <bool expr, class T = void>
	struct enable_if {}; // NOLINT(readability-identifier-naming)

	template <class T>
	struct enable_if<true, T> {
		using type = T;
	};

	// Replacement implementation of std::enable_if_t from C++14, included here for
	// interoperability with C++11
	template <bool B, class T = void>
	using enable_if_t = typename enable_if<B, T>::type;

	template <typename T>
	using is_vector = std::is_same<T, typename std::vector<typename T::value_type>>;

	template <typename T>
	struct remove_reference { // NOLINT(readability-identifier-naming)
		using type = T;
	};
	template <typename T>
	struct remove_reference<T&> {
		using type = T;
	};
	template <typename T>
	struct remove_reference<T&&> {
		using type = T;
	};

	template <typename T, typename U>
	inline enable_if_t<std::is_integral<T>::value && (std::is_integral<U>::value || std::is_enum<U>::value), typename remove_reference<T>::type> div_up(const T a, const U b)
	{
		assert(b > 0);
		assert(a >= 0);
		if (a <= 0) return 0;
		return static_cast<typename remove_reference<T>::type>(1 + (a - 1) / b);
	}

	inline int dnnl_get_max_threads() 
	{
		return omp_get_max_threads();
	}

	inline int dnnl_in_parallel() 
	{
		return omp_in_parallel();
	}

	inline void dnnl_thr_barrier() 
	{
#pragma omp barrier
	}

	inline int dnnl_get_current_num_threads() 
	{
		if (dnnl_in_parallel()) return 1;
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
		return omp_get_max_threads();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
		return tbb::this_task_arena::max_concurrency();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
		using namespace dnnl::impl::threadpool_utils;
		dnnl::threadpool_interop::threadpool_iface* tp = get_active_threadpool();
		return (tp) ? dnnl_get_max_threads() : 1;
#else
		return 1;
#endif
	}

	/* general parallelization */
	inline int adjust_num_threads(int nthr, std::size_t work_amount) 
	{
		if (nthr == 0) nthr = dnnl_get_current_num_threads();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
		return (work_amount == 1 || omp_in_parallel()) ? 1 : nthr;
#else
		return (int)std::min((dim_t)nthr, work_amount);
#endif
	}

	template <typename T, typename U>
	inline void balance211(T n, U team, U tid, T& n_start, T& n_end) {
		T n_min = 1;
		T& n_my = n_end;
		if (team <= 1 || n == 0) {
			n_start = 0;
			n_my = n;
		}
		else if (n_min == 1) {
			// team = T1 + T2
			// n = T1*n1 + T2*n2  (n1 - n2 = 1)
			T n1 = div_up(n, (T)team);
			T n2 = n1 - 1;
			T T1 = n - n2 * (T)team;
			n_my = (T)tid < T1 ? n1 : n2;
			n_start = (T)tid <= T1 ? (T)tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
		}

		n_end += n_start;
	}

	static inline void for_nd(const int ithr, const int nthr, std::size_t D0, const std::function<void(std::size_t)>& f)
	{
		std::size_t start{ 0 }, end{ 0 };
		balance211(D0, nthr, ithr, start, end);
		for (auto d0 = start; d0 < end; ++d0)
			f(d0);
	}

	static inline void parallel(int nthr, const std::function<void(int, int)>& f) 
	{
		nthr = adjust_num_threads(nthr, INT64_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
		for (int i = 0; i < nthr; ++i) {
			f(i, nthr);
		}
#else
#if defined(DNNL_ENABLE_ITT_TASKS)
		auto task_primitive_kind = itt::primitive_task_get_current_kind();
		auto task_primitive_info = itt::primitive_task_get_current_info();
		auto task_primitive_log_kind = itt::primitive_task_get_current_log_kind();
		auto task_primitive_itt_id = itt::primitive_task_get_itt_id();
		bool itt_enable = itt::get_itt(itt::__itt_task_level_high);
#endif
#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
		// Tasks must be always submitted to a threadpool, it will handle them
		// properly.
		if (nthr == 1) 
		{
			f(0, 1);
			return;
		}
#endif
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
		{
			int nthr_ = omp_get_num_threads();
			int ithr_ = omp_get_thread_num();
			assert(nthr_ == nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
			if (ithr_ && itt_enable) {
				itt::primitive_task_start(
					task_primitive_kind, task_primitive_log_kind);
				itt::primitive_add_metadata_and_id(task_primitive_info,
					task_primitive_log_kind, task_primitive_itt_id);
			}
#endif
			f(ithr_, nthr_);
#if defined(DNNL_ENABLE_ITT_TASKS)
			if (ithr_ && itt_enable)
				itt::primitive_task_end(task_primitive_log_kind);
#endif
		}
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
		tbb::parallel_for(0, nthr, [&](int ithr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
			bool mark_task = itt::primitive_task_get_current_kind()
				== primitive_kind::undefined;
			if (mark_task && itt_enable) {
				itt::primitive_task_start(
					task_primitive_kind, task_primitive_log_kind);
				itt::primitive_add_metadata_and_id(task_primitive_info,
					task_primitive_log_kind, task_primitive_itt_id);
			}
#endif
			f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
			if (mark_task && itt_enable)
				itt::primitive_task_end(task_primitive_log_kind);
#endif
			}, tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
		using namespace dnnl::impl::threadpool_utils;
		dnnl::threadpool_interop::threadpool_iface* tp = get_active_threadpool();
		if (!tp || dnnl_in_parallel()) {
			threadpool_utils::deactivate_threadpool();
			for (int ithr = 0; ithr < nthr; ithr++) {
				f(ithr, nthr);
			}
			threadpool_utils::activate_threadpool(tp);
		}
		else {
			tp->parallel_for(nthr, [=](int ithr, int nthr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
				bool is_master = threadpool_utils::get_active_threadpool() == tp;
				if (!is_master && itt_enable) {
					itt::primitive_task_start(
						task_primitive_kind, task_primitive_log_kind);
					itt::primitive_add_metadata_and_id(task_primitive_info,
						task_primitive_log_kind, task_primitive_itt_id);
				}
#endif
				f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
				if (!is_master && itt_enable) {
					itt::primitive_task_end(task_primitive_log_kind);
				}
#endif
				});
		}
#endif
#endif
	}

	static inline void parallel_nd(std::size_t D0, const std::function<void(std::size_t)>& f)
	{
		int nthr = adjust_num_threads(omp_get_max_threads(), D0);
		if (nthr)
			parallel(nthr, [=](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
	}
	

	static inline void parallel_nd(std::size_t D0, std::size_t threads, const std::function<void(std::size_t)>& f)
	{
		int nthr = std::min(static_cast<std::size_t>(adjust_num_threads(omp_get_max_threads(), D0)), threads);
		//int nthr = adjust_num_threads(omp_get_max_threads(), D0);
		if (nthr)
			parallel(nthr, [=](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
	}

	template <typename Func>
	inline void for_i(const size_t range, const Func& f)
	{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
		
		parallel_nd(range, [=](size_t i)
		{
	  		f(i);
		});
#else
		for_(0ull, range, [&](const blocked_range& r)
		{
			for (auto i = r.begin(); i < r.end(); i++)
				f(i);
		});
#endif
	}


	template <typename Func>
	inline void for_i(const size_t range, const size_t threads, const Func& f)
	{
		if (std::min(range, threads) > 1)
		{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
			
			parallel_nd(range, threads, [=](size_t i)
			{
	  			f(i);
			});
#else
			for_(0ull, range, [&](const blocked_range& r)
			{
				for (auto i = r.begin(); i < r.end(); i++)
					f(i);
			});
#endif
		}
		else
			for (auto i = 0ull; i < range; i++)
				f(i);
	}

	void fast_memzero(void *dest, size_t numbytes)
	{
  		const auto PAGE_4K = 256ull * 1024ull * 1024ull;
		const auto res = std::lldiv(static_cast<long long>(numbytes), static_cast<long long>(PAGE_4K));
		
  		if (!res.quot)
	  		fast_memset(dest, 0, res.rem);
  		else
			for_i(res.quot, [=](size_t i)
			{
      			const auto tail = (i + 1 == res.quot) ? res.rem : 0;
      			const auto ptr = reinterpret_cast<unsigned char *>(dest) + i * PAGE_4K;
				fast_memset(ptr, 0, PAGE_4K + tail);
    		});
	}
}