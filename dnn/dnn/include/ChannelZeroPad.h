#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelZeroPad final : public Layer
	{
	public:
		ChannelZeroPad(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt c) :
			Layer(device, format, name, LayerTypes::ChannelZeroPad, 0, 0, c, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() == 1);
			assert(InputLayer->C >= 1);
			assert(InputLayer->C < C);

			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(10);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
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
			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (NeuronsFormat == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetMemoryFormat(*InputLayer->DstMemDesc);
					if (ChosenFormat != GetMemoryFormat(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			assert(ChosenFormat == GetMemoryFormat(*InputLayer->DstMemDesc));
			if (ChosenFormat != GetMemoryFormat(*InputLayer->DstMemDesc))
				throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + InputLayer->Name);
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			
			if (!training)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						for (auto c = 0ull; c < InputLayer->PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)];

						for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
						{
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
						}
					}
					else
					{
						for (auto c = 0ull; c < InputLayer->C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPlainMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPlainMem(0, c, h, w)];

						for (auto c = InputLayer->C; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPlainMem(0, c, h, w)] = Float(0);
					}
				}
				else
				{
#endif
					const auto threads = GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);

					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)];

							for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);

						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPlainMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPlainMem(n, c, h, w)];

							for (auto c = InputLayer->C; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPlainMem(n, c, h, w)] = Float(0);
						});
			
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						for (auto c = 0ull; c < InputLayer->PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)];
#ifndef DNN_LEAN
									NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
								}
#endif // DNN_LEAN

						for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
#ifndef DNN_LEAN
									NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif // DNN_LEAN
								}
					}
					else
					{
						for (auto c = 0ull; c < InputLayer->C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPlainMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPlainMem(0, c, h, w)];
#ifndef DNN_LEAN
									NeuronsD1[OffsetPlainMem(0, c, h, w)] = Float(0);
#endif // DNN_LEAN
								}

						
						for (auto c = InputLayer->C; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPlainMem(0, c, h, w)] = Float(0);
#ifndef DNN_LEAN
									NeuronsD1[OffsetPlainMem(0, c, h, w)] = Float(0);
#endif // DNN_LEAN
								}
					}
				}
				else
				{
#endif
					const auto threads = GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)];
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
									}
#endif // DNN_LEAN
							for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif // DNN_LEAN		
									}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPlainMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPlainMem(n, c, h, w)];
#ifndef DNN_LEAN
										NeuronsD1[OffsetPlainMem(n, c, h, w)] = Float(0);
#endif // DNN_LEAN
									}

							for (auto c = InputLayer->C; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPlainMem(n, c, h, w)] = Float(0);
#ifndef DNN_LEAN
										NeuronsD1[OffsetPlainMem(n, c, h, w)] = Float(0);
#endif // DNN_LEAN
									}
						});
			
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
					for (auto c = 0ull; c < InputLayerBwd->PaddedC; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, c, h, w)];
				else
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							PRAGMA_OMP_SIMD()
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPlainMem(0, c, h, w)] += NeuronsD1[OffsetPlainMem(0, c, h, w)];
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPlainMem(n, c, h, w)] += NeuronsD1[OffsetPlainMem(n, c, h, w)];
					});
#ifdef DNN_STOCHASTIC
			}
#endif

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}