#pragma once
#include "Layer.h"

namespace dnn
{
	class Dropout final : public Layer
	{
	public:
		const bool LocalValue;
		Float Keep;
		Float Scale;
		FloatArray NeuronsActive;

		Dropout(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.5), const bool localValue = false) :
			Layer(device, format, name, LayerTypes::Dropout, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, false, false, dropout > 0),
			LocalValue(localValue),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout)),
			NeuronsActive(FloatArray())
		{
			assert(Inputs.size() == 1);

			FwdInferenceWeight = Float(2);
			FwdTrainingWeight = Float(4);
			BwdTrainingWeight = Float(4);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		void UpdateDropout(const Float dropout)
		{
			if (!LocalValue)
			{
				Enabled = dropout > 0;
				Keep = Float(1) - dropout;
				Scale = Float(1) / Keep;
			}
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Dropout:") + tab + FloatToString(Float(1) - Keep));
			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

			return description;
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			DNN_UNREF_PAR(batchSize);

			ChosenFormat = GetMemoryFormat(*InputLayer->DstMemDesc);

			DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);
		}

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if (Enabled)
			{
				NeuronsActive.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				const auto size = GetElementsCount();
				for (auto n = 0ull; n < batchSize; n++)
				{
					const auto offset = n * size;
					PRAGMA_OMP_SIMD()
					for (auto i = offset; i < offset + size; i++)
						NeuronsActive[i] = Float(1);
				}
			}
			else
				NeuronsActive.release();
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto size = GetElementsCount();
			const auto part = GetVectorPart(size);
			const auto strideHW = HW() * VectorSize;

			if (Enabled && training)
			{				
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);
				if (plain)
					for_i(batchSize, threads, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						VecFloat mask;
						for (auto i = start; i < end; i += VectorSize)
						{
							mask = BernoulliVecFloat(Keep);
							mask.store_a(&NeuronsActive[i]);
							(mask * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
#ifndef DNN_LEAN
							VecZero.store_nt(&NeuronsD1[i]);
#endif
						}
						for (auto i = end; i < start + size; i++)
						{
							NeuronsActive[i] = Bernoulli<Float>(Keep);
							Neurons[i] = NeuronsActive[i] * Scale * InputLayer->Neurons[i];
#ifndef DNN_LEAN
							NeuronsD1[i] = Float(0);
#endif
						}
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat mask;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
							for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
							{
								mask = BernoulliVecFloat(Keep);
								mask.store_a(&NeuronsActive[hw]);
								(mask * Scale * VecFloat().load_a(&InputLayer->Neurons[hw])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
								VecZero.store_nt(&NeuronsD1[hw]);
#endif
							}
						}
					});
			}
			else
			{
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);

				if (plain)
					for_i(batchSize, threads, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						for (auto i = start; i < end; i += VectorSize)
							(VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
						for (auto i = end; i < start + size; i++)
							Neurons[i] = InputLayer->Neurons[i];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
							for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
								(VecFloat().load_a(&InputLayer->Neurons[hw])).store_a(&Neurons[hw]);
							
						}
					});
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
			
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif
			const auto plain = IsPlainFormat();
			const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
			const auto strideHW = HW() * VectorSize;

			if (Enabled)
			{
				if (plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
							PRAGMA_OMP_SIMD()
							for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
								InputLayerBwd->NeuronsD1[hw] += NeuronsActive[hw] * NeuronsD1[hw];
						}
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
							for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
								mul_add(VecFloat().load_a(&NeuronsActive[hw]), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
						}
					});
			}
			else
			{
				if (plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
							PRAGMA_OMP_SIMD()
							for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
								InputLayerBwd->NeuronsD1[hw] += NeuronsD1[hw];
						}
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
							for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
								(VecFloat().load_a(&NeuronsD1[hw]) + VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
						}
					});
			}
#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}

		UInt GetNeuronsSize(const UInt batchSize) const override
		{
			return Layer::GetNeuronsSize(batchSize) + (batchSize * PaddedCDHW() * sizeof(Float));
		}
	};
}