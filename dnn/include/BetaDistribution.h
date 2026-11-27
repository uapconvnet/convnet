#pragma once
#include <random>
#include <limits>

namespace dnn
{
    /* https://stackoverflow.com/questions/15165202/random-number-generator-with-beta-distribution */
	template <typename RealType = double>
	class BetaDistribution
	{
	public:
		typedef RealType result_type;

		class param_type
		{
		public:
			typedef BetaDistribution distribution_type;

			explicit param_type(RealType a = 2.0, RealType b = 2.0) : a_param(a), b_param(b) { }

			RealType a() const noexcept { return a_param; }
			RealType b() const noexcept { return b_param; }

			bool operator==(const param_type& other) const noexcept
			{
				return (a_param == other.a_param && b_param == other.b_param);
			}

			bool operator!=(const param_type& other) const noexcept
			{
				return !(*this == other);
			}

		private:
			RealType a_param, b_param;
		};

		explicit BetaDistribution(RealType a = 2.0, RealType b = 2.0) noexcept : a_param(a), b_param(b) { }
		explicit BetaDistribution(const param_type& param) noexcept : a_param(param.a()), b_param(param.b()) { }

		void reset() noexcept { }

		param_type param() const noexcept
		{
			return param_type(a_param, b_param);
		}

		void param(const param_type& param) noexcept
		{
			a_param = param.a();
			b_param = param.b();
			cache_valid = false;
		}

		template <typename URNG>
		inline result_type operator()(URNG& engine) noexcept
		{
			return generate(engine);
		}

		template <typename URNG>
		inline result_type operator()(URNG& engine, const param_type& param) noexcept
		{
			BetaDistribution dist(param);
			return dist.generate(engine);
		}

		result_type min() const noexcept { return 0.0; }
		result_type max() const noexcept { return 1.0; }

		result_type a() const noexcept { return a_param; }
		result_type b() const noexcept { return b_param; }

		bool operator==(const BetaDistribution& other) const noexcept
		{
			return a_param == other.a_param && b_param == other.b_param;
		}

		bool operator!=(const BetaDistribution& other) const noexcept
		{
			return !(*this == other);
		}

	private:
		RealType a_param, b_param;
		
		typedef std::gamma_distribution<RealType> GammaDistType;
		mutable GammaDistType x_gamma_cached;
        mutable GammaDistType y_gamma_cached;
        mutable bool cache_valid = false;
	
		template <typename URNG>
		inline result_type generate(URNG& engine) noexcept
		{
    		// Validate parameters first
			if (a_param <= 0.0 || b_param <= 0.0)
				return 0.5;

    		// Update cached distributions only when parameters change
			if (!cache_valid)
			{
				x_gamma_cached = GammaDistType(a_param);
				y_gamma_cached = GammaDistType(b_param);
				cache_valid = true;
			}
    
			result_type x = x_gamma_cached(engine);
			result_type y = y_gamma_cached(engine);
    
			const result_type sum = x + y;
    
			if (sum <= std::numeric_limits<result_type>::min())
				return 0.0;
    
			result_type ratio = x / sum;
    		// Clamp to valid range
			return (ratio < 0.0) ? 0.0 : ((ratio > 1.0) ? 1.0 : ratio);
		}
	};

	// Standard library naming convention alias for compatibility
	template <typename RealType = double>
	using beta_distribution = BetaDistribution<RealType>;
}