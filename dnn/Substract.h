#pragma once
#include "Layer.h"

namespace dnn
{
	class Substract final : public Layer
	{
	private:
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		
	public:
		const Byte first, second;
		
		Substract(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Substract, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);

			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(5);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}

		void UpdateResolution() final override
		{
			H = Inputs[first]->H;
			W = Inputs[first]->W;
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
			if (GetMemoryNDims(*Inputs[first]->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (NeuronsFormat == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetMemoryFormat(*Inputs[first]->DstMemDesc);
					if (ChosenFormat != GetMemoryFormat(*Inputs[first]->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_sub, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (training)
			{
				if (Reference)
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();
#ifndef DNN_LEAN
					InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif // DNN_LEAN
				}
				else
				{
					const auto plain = IsPlainFormat();
					const auto size = GetElementsCount();
					const auto part = GetVectorPart(size);
					const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * size, FwdTrainingWeight);
					const auto strideHW = HW() * VectorSize;

					if (plain)
					{
						if (EqualDimensions(Inputs))
						{
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * CDHW();
								const auto end = start + CDHW();
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] - Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = 0;
#endif
								}
							});
						}
						else
						{
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = n * CDHW() + c * HW();
									const auto channelOffset = n * C + c;
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Neurons[hw + outputOffset] = Inputs[first]->Neurons[hw + outputOffset] - Inputs[second]->Neurons[channelOffset];
#ifndef DNN_LEAN
										NeuronsD1[hw + outputOffset] = 0;
#endif
									}
								}
							});
						}
					}
					else
					{
						if (EqualDimensions(Inputs))
						{
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;
								for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
								{
									(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) - VecFloat().load_a(&Inputs[1]->Neurons[cdhw])).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
								}
								for (auto cdhw = start + part; cdhw < start + size; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] - Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = 0;
#endif							
								}
							});						
						}
						else
						{
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = n * PaddedCDHW() + c * HW();
									const auto channelOffset = n * PaddedC + c;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) - VecFloat().load_a(&Inputs[second]->Neurons[channelOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
									}
								}
							});
						}
					}
				}
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto plain = IsPlainFormat();
			const auto size = GetElementsCount();
			const auto part = GetVectorPart(size);

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (EqualDimensions(Inputs))
				{
					if (plain)
					{
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					}
					else
					{
						VecFloat inputD1, D1;
						for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&InputsBwd[0]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&InputsBwd[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&InputsBwd[1]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
		}
						for (auto cdhw = part; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					}
				}
				else
				{
					if (plain)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = c * HW();
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
							{
								InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset];
								InputsBwd[second]->NeuronsD1[c] += NeuronsD1[hw + outputOffset];
							}
						}
					}
					else
					{
						const auto strideHW = HW() * VectorSize;
						VecFloat D1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								D1.load_a(&NeuronsD1[hw + outputOffset]);
								(D1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset])).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
								(D1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[c])).store_a(&InputsBwd[second]->NeuronsD1[c]);
							}
						}
					}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * size, BwdTrainingWeight);

				if (EqualDimensions(Inputs))
				{
					if (plain)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							const auto end = start + size;
							PRAGMA_OMP_SIMD()
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
								InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							}
						});	
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;

							VecFloat D1;
							for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);
								(VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw]) + D1).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
								(VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw]) + D1).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
							}
							for (auto cdhw = start + part; cdhw < start + size; cdhw++)
							{
								InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
								InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							}
						});
					}
				}
				else
				{
					if (plain)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto outputOffset = n * CDHW() + c * HW();
								const auto channelOffset = n * C + c;
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset];
									InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset];
								}
							}
						});
					}
					else
					{
						const auto strideHW = HW() * VectorSize;

						for_i(batchSize, threads, [=](UInt n)
						{
							VecFloat D1;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto outputOffset = n * PaddedCDHW() + c * HW();
								const auto channelOffset = n * PaddedC + c;
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									D1.load_a(&NeuronsD1[hw + outputOffset]);
									(D1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset])).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
									(D1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
								}
							}
						});
					}
				}
#ifdef DNN_STOCHASTIC
			}
#endif

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}