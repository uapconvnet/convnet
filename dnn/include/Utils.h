#pragma once

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#include "stdafx.h"
#else
#include <sys/sysinfo.h>
#endif

#ifdef NDEBUG
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#ifndef MAX_VECTOR_SIZE
#ifdef DNN_AVX2
#define INSTRSET 8
#define MAX_VECTOR_SIZE 256
#endif //DNN_AVX2

#ifdef DNN_AVX512
#define INSTRSET 10
#define MAX_VECTOR_SIZE 512
#endif //DNN_AVX512

#ifdef DNN_AVX512BW
#define INSTRSET 10
#define MAX_VECTOR_SIZE 512
#endif //DNN_AVX512BW

#ifdef DNN_AVX512FP16
#define INSTRSET 12
#define MAX_VECTOR_SIZE 512
#endif //DNN_AVX512FP16
#endif // MAX_VECTOR_SIZE

#include "instrset.h"
#include "vectorclass.h"
#include "vectormath_common.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"
#include "add-on/random/ranvec1.h"

#include <algorithm>
#include <array>
#include <atomic>
//#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <execution>
#include <filesystem>
#include <functional> 
#include <future>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <locale>
#include <clocale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

#ifndef NDEBUG
#ifdef _WIN32
#pragma fenv_access (on)
#else
#pragma STDC FENV_ACCESS ON
#endif
#endif

#include "dnnl.hpp"
#include "dnnl_debug.h"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#include "fastmem.h"
#include "ParallelFor.h"
#include "AlignedAllocator.h"
#include "AlignedArray.h"
#include "AlignedMemory.h"
#include "BetaDistribution.h"

#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 255
#include "magic_enum/magic_enum.hpp"

#include "bitsery/bitsery.h"
#include "bitsery/adapter/stream.h"
#include "bitsery/traits/string.h"
#include "bitsery/traits/vector.h"
// #include "bitsery/ext/std_atomic.h"
// #include "bitsery/ext/growable.h"


namespace dnn
{
	constexpr auto DefaultDatasetMeanStdDev = false;
	constexpr auto Inplace = true;
	constexpr auto Kahan = true;
	constexpr auto PlainOptimizerWeights = true;
	constexpr auto SingleMeanVariancePass = true;

	constexpr auto TestActivations = false;
	constexpr auto TestBatchNormalization = false;
	constexpr auto TestConcat = false;
	constexpr auto TestMultiply = false;
	constexpr auto TestReduction = false;

	constexpr auto Reference = false;
	constexpr auto ReferenceAdd = false;
	constexpr auto ReferenceBatchNormalization = false;
	constexpr auto ReferenceConcat = false;
	constexpr auto ReferenceMultiply = false;
	constexpr auto ReferenceReduction = false;
		
	typedef float Float;
	typedef double Double;
	typedef long long Int;
	typedef std::size_t UInt;
	typedef unsigned char Byte;
	
    typedef AlignedMemory<Float> FloatArray;
	typedef AlignedArray<Byte, 64ull> ByteArray;
	typedef std::vector<Float, AlignedAllocator<Float, 64ull>> FloatVector;

	//constexpr bool IsLittleEndian = std::endian::native == std::endian::little;
	constexpr auto WeightsLimit = Float(500);	// limit for all the weights and biases [-WeightsLimit,WeightsLimit]
	constexpr auto PlainFmt = dnnl::memory::format_tag::abcd;
	
	auto GetThreads(const UInt elements, const Float weight = Float(1)) NOEXCEPT
	{
		static const auto maxThreads = static_cast<UInt>(omp_get_max_threads());

		constexpr auto ultraLightThreshold =   2097152ull;
		constexpr auto lightThreshold =        8338608ull;
		constexpr auto mediumThreshold =      68338608ull;
		constexpr auto heavyThreshold =      120338608ull;
		constexpr auto maximumThreshold =    187538608ull;
		
		const auto ultraLight = 2ull;
		const auto light      = maxThreads >=  6ull ?  4ull : 2ull;
		const auto medium     = maxThreads >=  8ull ?  8ull : maxThreads >=  6ull ?  6ull : maxThreads >=  4ull ?  4ull : 2ull;
		const auto heavy      = maxThreads >= 32ull ? 16ull : maxThreads >= 24ull ?  16ll : maxThreads >= 16ull ? 16ull : maxThreads >= 12ull ? 12ull : maxThreads >= 8ull ? 8ull : maxThreads >= 6ull ? 6ull : maxThreads >= 4ull ? 4ull : 2ull;
		const auto ultraHeavy = maxThreads >= 32ull ? 32ull : maxThreads >= 24ull ? 24ull : maxThreads >= 16ull ? 16ull : maxThreads >= 12ull ? 12ull : maxThreads >= 8ull ? 8ull : maxThreads >= 6ull ? 6ull : maxThreads >= 4ull ? 4ull : 2ull;

		const auto load = static_cast<UInt>(Float(elements) * weight);

		return
			load < ultraLightThreshold ? ultraLight :
			load < lightThreshold ?           light :
			load < mediumThreshold ?         medium :
			load < heavyThreshold ?           heavy :
			load < maximumThreshold ?    ultraHeavy : maxThreads;
	}
	
	
#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
	typedef Vec16f VecFloat;
	typedef Vec16fb VecFloatBool;
	constexpr auto VectorSize = 16ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw16c;
#elif defined(DNN_AVX2)
	typedef Vec8f VecFloat;
	typedef Vec8fb VecFloatBool;
	constexpr auto VectorSize = 8ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw8c;
#endif
	const auto VecZero = VecFloat(Float(0));
	
	/*
	static inline int div_up(int value, int divisor) {	return (value + divisor - 1) / divisor;	}
	// Round value down to a multiple of factor.
	static inline int align_down(int value, int factor)	{ return factor * (value / factor);	}
	// Round value up to a multiple of factor.
	static inline int align_up(int value, int factor) {	return factor * div_up(value, factor); }
	*/

	template<typename T>
	static DNN_INLINE constexpr auto Square(const T& value) NOEXCEPT { return (value * value); }
	template <typename T>
	static DNN_INLINE constexpr auto Clamp(T val, T lo, T hi) NOEXCEPT { return std::min<T>(hi, std::max<T>(lo, val)); }
	template<typename T>
	static DNN_INLINE constexpr auto Saturate(const T& value) NOEXCEPT { return (value > T(255)) ? Byte(255) : (value < T(0)) ? Byte(0) : Byte(value); }
	template<typename T>
	static DNN_INLINE constexpr auto GetColorFromRange(const T& range, const T& minimum, const T& value) NOEXCEPT { return Saturate<T>(T(255) - ((value - minimum) * range)); }
	template<typename T>
	static DNN_INLINE constexpr auto GetColorRange(const T& min, const T& max) NOEXCEPT { return (min == max) ? T(0) : T(255) / ((std::signbit(min) && std::signbit(max)) ? -(min + max) : (max - min)); }
	static auto DNN_INLINE ClampVecFloat(const VecFloat& v, const Float& lo, const Float& hi) NOEXCEPT { return min(VecFloat(hi), max(v, VecFloat(lo))); }
	
	constexpr auto GetVectorPart(const UInt& elements) NOEXCEPT { return (elements / VectorSize) * VectorSize; }
	constexpr auto DivUp(const UInt& c) NOEXCEPT { if (c == 0ull) return 0ull; else return (((c - 1) / VectorSize) + 1) * VectorSize; }
	
	auto IsPlainDataFmt(const dnnl::memory::desc& md) NOEXCEPT { return md.get_format_kind() == dnnl::memory::format_kind::blocked && md.get_inner_nblks() == 0; }
	auto GetMemoryNDims(const dnnl::memory::desc& md) NOEXCEPT
	{
		if (!md.is_zero())
			return md.get_ndims();

		return 0;
	}
	auto GetMemoryFormat(const dnnl::memory::desc& md) NOEXCEPT
	{
		using format_tag = dnnl::memory::format_tag;

		if (!md.is_zero())
		{
			//const auto data_type = md.get_data_type();
			const auto dims = md.get_dims();
			const auto format_kind = md.get_format_kind();
			const auto inner_blks = md.get_inner_blks();
			const auto inner_idxs = md.get_inner_idxs();
			const auto inner_nblks = md.get_inner_nblks();
			const auto ndims = md.get_ndims();
			const auto padded_dims = md.get_padded_dims();
			const auto padded_offsets = md.get_padded_offsets();
			const auto strides = md.get_strides();

			if (format_kind == dnnl::memory::format_kind::blocked)
			{
				if (inner_nblks == 0)
				{
					if (ndims == 1)
					{
						return format_tag::a;
					}
					if (ndims == 2)
					{

						if (strides[0] == 1)
							return format_tag::ba;
						else if (strides[1] == 1)
							return format_tag::ab;
					}
					if (ndims == 3)
					{
						if (strides[0] == 1)
							return format_tag::bca;
						else if (strides[1] == 1)
							return format_tag::acb;
						else if (strides[2] == 1)
							return format_tag::abc;
					}
					if (ndims == 4)
					{
						if (strides[0] == 1)
							return format_tag::bcda;
						else if (strides[1] == 1)
							return format_tag::acdb;
						else if (strides[2] == 1)
							return format_tag::abdc;
						else if (strides[3] == 1)
							return format_tag::abcd;
					}
					if (ndims == 5)
					{
						if (strides[0] == 1)
							return format_tag::bcdea;
						else if (strides[1] == 1)
							return format_tag::acdeb;
						else if (strides[2] == 1)
							return format_tag::abdec;
						else if (strides[3] == 1)
							return format_tag::abced;
						else if (strides[4] == 1)
							return format_tag::abcde;
					}
				}
				else
				{
					if (ndims == 2)
					{
						return format_tag::ab;
					}
					if (ndims == 3)
					{
						return format_tag::abc;
					}
					if (ndims == 4)
					{
						if (inner_nblks == 1 && inner_idxs[0] == 0)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::Abcd4a;
							case 8:
								return format_tag::Abcd8a;
							case 16:
								return format_tag::Abcd16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 1 && inner_idxs[0] == 1)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::aBcd4b;
							case 8:
								return format_tag::aBcd8b;
							case 16:
								return format_tag::aBcd16b;
							default:
								return format_tag::undef;
							}
						}
						//else if (inner_nblks == 1 && inner_idxs[0] == 2)
						//{
						//	switch (inner_blks[0])
						//	{
						//	case 4:
						//		return format_tag::abCd4c;
						//	/*case 8:
						//		return format_tag::abCd4a8c;
						//	case 16:
						//		return format_tag::abCd16c;*/
						//	default:
						//		return format_tag::undef;
						//	}
						//}
						else if (inner_nblks == 2 && inner_idxs[0] == 1 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcd4b4a;
							case 8:
								return format_tag::ABcd8b8a;
							case 16:
								return format_tag::ABcd16b16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 0 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcd4a4b;
							case 8:
								return format_tag::ABcd8a8b;
							case 16:
								return format_tag::ABcd16a16b;
							default:
								return format_tag::undef;
							}
						}
					}
					if (ndims == 5)
					{
						if (inner_nblks == 1 && inner_idxs[0] == 0)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::Abcde4a;
							case 8:
								return format_tag::Abcde8a;
							case 16:
								return format_tag::Abcde16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 1 && inner_idxs[0] == 1)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::aBcde4b;
							case 8:
								return format_tag::aBcde8b;
							case 16:
								return format_tag::aBcde16b;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 1 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcde4b4a;
							case 8:
								return format_tag::ABcde8b8a;
							case 16:
								return format_tag::ABcde16b16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 0 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcde4a4b;
							case 8:
								return format_tag::ABcde8a8b;
							case 16:
								return format_tag::ABcde16a16b;
							default:
								return format_tag::undef;
							}
						}

						/*

						/// 5D tensor blocked by 1st dimension with block size 16
						dnnl_ABcde4b16a4b,
						/// 5D tensor blocked by 1st dimension with block size 8
						dnnl_ABcde2b8a4b,
						/// 5D tensor blocked by 2nd dimension with block size 16
						dnnl_aBcde16b,
						dnnl_ABcde16b16a,
						dnnl_aBCde16b16c,
						dnnl_aBCde16c16b,
						dnnl_aBCde2c8b4c,
						dnnl_Abcde4a,
						/// 5D tensor blocked by 2nd dimension with block size 32
						dnnl_aBcde32b,
						/// 5D tensor blocked by 2nd dimension with block size 4
						dnnl_aBcde4b,
						dnnl_ABcde4b4a,
						dnnl_ABcde4a4b,
						dnnl_aBCde4b4c,
						dnnl_aBCde2c4b2c,
						dnnl_aBCde4b8c2b,
						dnnl_aBCde4c16b4c,
						dnnl_aBCde16c16b4c,
						dnnl_aBCde16c16b2c,
						dnnl_aBCde4c4b,
						dnnl_Abcde8a,
						dnnl_ABcde8a8b,
						dnnl_ABcde8a4b,
						dnnl_BAcde16b16a,
						/// 5D tensor blocked by 2nd dimension with block size 8
						dnnl_aBcde8b,
						dnnl_ABcde8b16a2b,
						dnnl_aBCde8b16c2b,
						dnnl_aBCde4c8b2c,
						dnnl_aCBde8b16c2b,
						dnnl_ABcde8b8a,
						dnnl_ABcde32a32b,
						dnnl_aBCde8b8c,
						dnnl_aBCde8b4c,
						dnnl_ABc4a8b8a4b,
						dnnl_ABcd4a8b8a4b,
						dnnl_ABcde4a8b8a4b,
						dnnl_BAc4b8a8b4a,
						dnnl_BAcd4b8a8b4a,
						dnnl_BAcde4b8a8b4a,
						dnnl_ABcd2a8b8a2b,
						dnnl_aBCd4b8c8b4c,
						dnnl_aBCde4b8c8b4c,
						dnnl_aBCde2b8c8b2c,
						dnnl_aBCde8c16b2c,
						dnnl_aBCde8c8b,
						/// 5D tensor blocked by 3rd dimension with block size 4
						dnnl_aBCde2b4c2b,

						*/
					}
				}
			}
		}

		return format_tag::undef;
	}

	/* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
	template<typename T>
	static void KahanSum(const T& value, T& sum, T& correction) NOEXCEPT
	{
		if constexpr (Kahan)
		{
			const auto y = value - correction;
			const auto t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		else
			sum += value;
	}

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
	const auto nwl = std::string("\r\n");
#elif defined(__APPLE__)
	const auto nwl = std::string("\r");
#else // assuming Linux
	const auto nwl = std::string("\n");
#endif
	const auto tab = std::string("\t");
	const auto dtab = std::string("\t\t");

#ifdef DNN_FAST_SEED
	template<typename T>
	static T Seed() NOEXCEPT
	{
		return static_cast<T>(__rdtsc());
	}
#else
	template<typename T>
	static T Seed() NOEXCEPT
	{
		return static_cast<T>(physicalSeed());
	}
#endif

	static auto BernoulliVecFloat(const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in BernoulliVecFloat function");
#endif
		static thread_local auto generator = Ranvec1(3, Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));

#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return select(generator.random16f() < p, VecFloat(Float(1)), VecFloat(Float(0)));
#elif defined(DNN_AVX2)
		return select(generator.random8f() < p, VecFloat(Float(1)), VecFloat(Float(0)));
#endif
	}

	static auto UniformVecFloat(const Float min = Float(0), const Float max = Float(1)) NOEXCEPT
	{
#ifndef NDEBUG
		if (min >= max)
			throw std::invalid_argument("Parameter out of range in UniformVecFloat function");
#endif
		static thread_local auto generator = Ranvec1(3, Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));
		const auto scale = std::abs(max - min);

#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return (generator.random16f() * scale) + min;
#elif defined(DNN_AVX2)
		return (generator.random8f() * scale) + min;
#endif
	}

	template<typename T>
	static auto Bernoulli(const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in Bernoulli function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return static_cast<T>(std::bernoulli_distribution(static_cast<double>(p))(generator));
	}

	template<typename T>
	static auto UniformInt(const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_integral<T>::value, "Only integral type supported in UniformInt function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformInt function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_int_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto UniformReal(const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in UniformReal function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformReal function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_real_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto TruncatedNormal(const T m, const T s, const T limit) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in TruncatedNormal function");
#ifndef NDEBUG
		if (limit < s)
			throw std::invalid_argument("limit out of range in TruncatedNormal function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		T x;
		do { x = std::normal_distribution<T>(T(0), s)(generator); } while (std::abs(x) > limit); // reject if beyond limit

		return x + m;
	}

	template<typename T>
	auto GetBetaDistribution(const T a, const T b) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in BetaDistribution function");
		static thread_local auto generator = std::mt19937(Seed<unsigned>());

		return dnn::BetaDistribution<T>(a, b)(generator);
	}

	struct no_separator : std::numpunct<char>
	{
	protected:
		virtual std::string do_grouping() const
		{
			return std::string("");
		}
		virtual char do_decimal_point() const
		{
			return ',';
		}
		virtual char do_thousands_sep() const
		{
			return '.';
		}
	};

	auto FloatToString(const Float value, const std::streamsize precision = 8)
	{
		auto ss = std::stringstream();
		ss.imbue(std::locale(std::locale(""), new no_separator()));
		ss.precision(precision);
		ss << value;
		return ss.str();
	}

	auto FloatToStringFixed(const Float value, const std::streamsize precision = 8)
	{
		auto ss = std::stringstream();
		ss.imbue(std::locale(std::locale(""), new no_separator()));
		ss.precision(precision);
		ss << std::fixed << value;
		return ss.str();
	}

	auto FloatToStringScientific(const Float value, const std::streamsize precision = 4)
	{
		auto ss = std::stringstream();
		ss.imbue(std::locale(std::locale(""), new no_separator()));
		ss.precision(precision);
		ss << std::scientific << value;
		return ss.str();
	}

	auto StringToFloat(const std::string& str, const std::locale& locale = std::locale(std::locale(""), new no_separator()))
	{
		auto value = Float(0);
		auto ss = std::stringstream(str);
		ss.imbue(locale);
		ss.precision(std::streamsize(8));
		ss >> std::defaultfloat >> value;
		return value;
	}

	auto GetFileSize(const std::string& fileName)
	{
		auto file = std::ifstream(fileName, std::ifstream::in | std::ifstream::binary);

		if (!file.is_open() || file.bad())
			return std::streamsize(-1);

		file.seekg(0, std::ios::beg);
		const auto start = file.tellg();
		file.seekg(0, std::ios::end);
		const auto end = file.tellg();
		file.close();
		
		return static_cast<std::streamsize>(end - start);
	}

	auto StringToLower(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		return text;
	}

	auto IsStringBool(const std::string& text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == std::string("true") || textLower == std::string("yes") || textLower == std::string("false") || textLower == std::string("no"))
			return true;

		return false;
	}

	auto StringToBool(const std::string& text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == std::string("true") || textLower == std::string("yes"))
			return true;

		return false;
	}

	auto BoolToString(const bool value)
	{
		return value ? std::string("Yes") : std::string("No");
	}

	auto GetTotalFreeMemory()
	{
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
		auto statusEx = MEMORYSTATUSEX();
		statusEx.dwLength = sizeof(MEMORYSTATUSEX);
		GlobalMemoryStatusEx(&statusEx);
		std::cout << std::string("Available memory: ") << std::to_string(statusEx.ullAvailPhys/1024/1024) << std::string("/") << std::to_string(statusEx.ullTotalPhys/1024/1024) << std::string(" MB") << std::endl;
		return statusEx.ullAvailPhys;
#else        
		struct sysinfo info;
		if (sysinfo(&info) == 0)
		{
			std::cout << std::string("Available memory: ") << std::to_string(info.freeram*info.mem_unit/1024/1024) << std::string("/") << std::to_string(info.totalram*info.mem_unit/1024/1024) << std::string(" MB") << std::endl;
			return static_cast<UInt>(info.freeram * info.mem_unit);
		}
		else
			return static_cast<UInt>(0);
#endif
	}
	
	auto CaseInsensitiveReplace(std::string::const_iterator begin, std::string::const_iterator end, const std::string& before, const std::string& after)
	{
		auto retval = std::string("");
		auto dest = std::back_insert_iterator<std::string>(retval);
		auto current = begin;
		auto next = std::search(current, end, before.begin(), before.end(), [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); });

		while (next != end)
		{
			std::copy(current, next, dest);
			std::copy(after.begin(), after.end(), dest);
			current = next + before.size();
			next = std::search(current, end, before.begin(), before.end(), [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); });
		}

		std::copy(current, next, dest);

		return retval;
	}

	/* https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring */
	auto Trim(std::string text)
	{
		text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](int ch) { return !std::isspace(ch); }));
		text.erase(std::find_if(text.rbegin(), text.rend(), [](int ch) { return !std::isspace(ch); }).base(), text.end());
		return text;
	}

	/* https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf */
	auto& SafeGetline(std::istream& is, std::string& line)
	{
		line.clear();

		// The characters in the stream are read one-by-one using a std::streambuf.
		// That is faster than reading them one-by-one using the std::istream.
		// Code that uses streambuf this way must be guarded by a sentry object.
		// The sentry object performs various tasks,
		// such as thread synchronization and updating the stream state.

		std::istream::sentry se(is, true);
		auto sb = is.rdbuf();

		for (;;) 
		{
			auto c = sb->sbumpc();
			switch (c) 
			{
			case '\n':
				return is;
			case '\r':
				if (sb->sgetc() == '\n')
					sb->sbumpc();
				return is;
			case std::streambuf::traits_type::eof():
				// Also handle the case when the last line has no line ending
				if (line.empty())
					is.setstate(std::ios::eofbit);
				return is;
			default:
				line += static_cast<char>(c);
			}
		}
	}

	template <typename T>
	constexpr void SwapEndian(T& buffer) NOEXCEPT
	{
		static_assert(std::is_standard_layout<T>::value, "SwapEndian supports standard layout types only");
		auto startIndex = static_cast<char*>((void*)buffer.data());
		auto endIndex = startIndex + sizeof(buffer);
		std::reverse(startIndex, endIndex);
	}

	/*
	auto RelativeError(const Float reference, const Float actual)
	{
		return std::abs(reference - actual) / std::max(std::numeric_limits<Float>().min(), std::abs(reference));
	}
	
	auto auto Median(FloatVector& array)
	{
		std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
		return array[array.size() / 2];
	}
	*/

	/* https://stackoverflow.com/questions/64169258/c-print-days-hours-minutes-etc-of-a-chronoduration */
	template <class Rep, std::intmax_t num, std::intmax_t denom>
	auto ChronoBurst(std::chrono::duration<Rep, std::ratio<num, denom>> d)
	{
		const auto hrs = std::chrono::duration_cast<std::chrono::hours>(d);
		const auto mins = std::chrono::duration_cast<std::chrono::minutes>(d - hrs);
		const auto secs = std::chrono::duration_cast<std::chrono::seconds>(d - hrs - mins);
		const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(d - hrs - mins - secs);

		return std::make_tuple(hrs, mins, secs, ms);
	}

	struct LabelInfo
	{
		UInt LabelA;
		UInt LabelB;
		Float Lambda;
	};
}
