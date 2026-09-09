#pragma once
#include <random>

namespace dnn
{
    /* https://stackoverflow.com/questions/15165202/random-number-generator-with-beta-distribution */
	template <typename RealType = double>
	class BetaDistribution
	{
	public:
		typedef RealType ResultType;

		class ParamType
		{
		public:
			typedef BetaDistribution distribution_type;

			explicit ParamType(RealType a = 2.0, RealType b = 2.0) : a_param(a), b_param(b) { }

			RealType a() const noexcept { return a_param; }
			RealType b() const noexcept { return b_param; }

			bool operator==(const ParamType& other) const noexcept
			{
				return (a_param == other.a_param && b_param == other.b_param);
			}

			bool operator!=(const ParamType& other) const noexcept
			{
				return !(*this == other);
			}

		private:
			RealType a_param, b_param;
		};

		explicit BetaDistribution(RealType a = 2.0, RealType b = 2.0) noexcept  : a_gamma(a), b_gamma(b) { }
		explicit BetaDistribution(const ParamType& param) noexcept : a_gamma(param.a()), b_gamma(param.b()) { }

		void reset() { }

		ParamType param() const noexcept
		{
			return ParamType(a(), b());
		}

		void param(const ParamType& param) noexcept
		{
			a_gamma = GammaDistType(param.a());
			b_gamma = GammaDistType(param.b());
		}

		template <typename URNG>
		inline ResultType operator()(URNG& engine) noexcept
		{
			return generate(engine, a_gamma, b_gamma);
		}

		template <typename URNG>
		inline ResultType operator()(URNG& engine, const ParamType& param) noexcept
		{
			GammaDistType a_param_gamma(param.a()), b_param_gamma(param.b());
			return generate(engine, a_param_gamma, b_param_gamma);
		}

		ResultType min() const noexcept { return 0.0; }
		ResultType max() const noexcept { return 1.0; }

		ResultType a() const noexcept { return a_gamma.alpha(); }
		ResultType b() const noexcept { return b_gamma.alpha(); }

		bool operator==(const BetaDistribution<ResultType>& other) const noexcept
		{
			return (param() == other.param() &&	a_gamma == other.a_gamma &&	b_gamma == other.b_gamma);
		}

		bool operator!=(const BetaDistribution<ResultType>& other) const noexcept
		{
			return !(*this == other);
		}

	private:
		typedef std::gamma_distribution<ResultType> GammaDistType;

		GammaDistType a_gamma, b_gamma;

		template <typename URNG>
		inline ResultType generate(URNG& engine, GammaDistType& x_gamma, GammaDistType& y_gamma) noexcept
		{
			ResultType x = x_gamma(engine);
			return x / (x + y_gamma(engine));
		}
	};
}
