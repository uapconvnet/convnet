#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelSplitRatioRight final : public Layer
	{
	private:
		std::unique_ptr<dnnl::memory::desc> MemDesc;

	public:
		const Float Ratio;
		const UInt ChannelsLeft;
		const bool Padded;
		
		ChannelSplitRatioRight(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float ratio = Float(0.375)) :
			Layer(device, format, name, LayerTypes::ChannelSplitRatioRight, 0, 0, UInt(std::roundf(Float(inputs[0]->C)) * ratio), inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Ratio(ratio),
			ChannelsLeft(UInt(std::roundf(Float(inputs[0]->C)) * (std::roundf(Float(1)) - ratio))),
			Padded(inputs[0]->C % VectorSize == 0 && C % VectorSize == 0 && UInt(std::roundf(Float(inputs[0]->C)) * (std::roundf(Float(1)) - ratio)) % VectorSize == 0)
		{
			assert(Inputs.size() == 1);
			assert(Ratio > Float(0));
			assert(Ratio < Float(1));

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
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Ratio:  ") + dtab + FloatToString(Ratio));

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
			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));

				if (Padded)
					MemDesc = std::make_unique<dnnl::memory::desc>(InputLayer->DstMemDesc->submemory_desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::dims({ dnnl::memory::dim(0), dnnl::memory::dim(ChannelsLeft) })));
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

				if (Padded)
					MemDesc = std::make_unique<dnnl::memory::desc>(InputLayer->DstMemDesc->submemory_desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::dims({ dnnl::memory::dim(0), dnnl::memory::dim(ChannelsLeft), dnnl::memory::dim(0), dnnl::memory::dim(0) })));

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
    	}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (Padded && !training)
			{
				const auto& memSrc = dnnl::memory(*MemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();

#ifndef DNN_LEAN
				/*if (training)
					InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW(), FwdZeroGradient);*/
#else
				DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN		
			}
			else
			{
				const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (training)
					{
						if (!plain)
						{
							for (auto c = 0ull; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)];
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif  // DNN_LEAN
									}

							for (auto c = C; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif  // DNN_LEAN
									}

						}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto inputOffset = (c + ChannelsLeft) * HW();
								const auto outputOffset = c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[hw + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
						}
					}
					else
					{
						if (!plain)
						{
							for (auto c = 0ull; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)];

							for (auto c = C; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto inputOffset = (c + ChannelsLeft) * HW();
								const auto outputOffset = c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
							}
						}
					}
				}
				else
				{
#endif
					if (training)
					{
						const auto threads = GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
										{
											Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)];
#ifndef DNN_LEAN
											NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif  // DNN_LEAN
										}

								for (auto c = C; c < PaddedC; c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
										{
											Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
#ifndef DNN_LEAN
											NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif  // DNN_LEAN
										}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto inputOffset = (n * InputLayer->CDHW()) + ((c + ChannelsLeft) * HW());
									const auto outputOffset = (n * CDHW()) + (c * HW());
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
										NeuronsD1[hw + outputOffset] = Float(0);
#endif // DNN_LEAN
									}
								}
							});
					}
					else
					{
						const auto threads = GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);

						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
											Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)];

								for (auto c = C; c < PaddedC; c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
											Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto inputOffset = (n * InputLayer->CDHW()) + ((c + ChannelsLeft) * HW());
									const auto outputOffset = (n * CDHW()) + (c * HW());
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
										Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
								}
							});
					}
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
				{
					for (auto c = 0ull; c < C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(0, c, h, w)];
				}
				else
					for (auto c = 0ull; c < C; c++)
					{
						const auto inputOffset = (c + ChannelsLeft) * HW();
						const auto outputOffset = c * HW();
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
					}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (n * InputLayerBwd->CDHW()) + ((c + ChannelsLeft) * HW());
							const auto outputOffset = (n * CDHW()) + (c * HW());
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
						}
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