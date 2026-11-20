#pragma once
#include "Layer.h"

namespace dnn
{
	class DropPathAdd final : public Layer
	{
	private:
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unordered_map<int, dnnl::memory> bwdArgsFirst;
        std::unordered_map<int, dnnl::memory> bwdArgsSecond;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
        std::unique_ptr<dnnl::binary::primitive_desc> bwdDescFirst;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdDescSecond;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
        std::unique_ptr<dnnl::binary> bwdFirst;
        std::unique_ptr<dnnl::binary> bwdSecond;
#endif
		std::vector<Float> scales;
		dnnl::memory scaleFirst;
		dnnl::memory scaleSecond;
		
	public:
		const Byte first, second;
		FloatVector SurvivalProbability;

		DropPathAdd(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::DropPathAdd, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			SurvivalProbability(FloatVector(2, Float(1))),
			scales(std::vector<Float>(2, Float(1))),
			scaleFirst(dnnl::memory::desc::host_scalar(dnnl::memory::data_type::f32), Float(1)),
			scaleSecond(dnnl::memory::desc::host_scalar(dnnl::memory::data_type::f32), Float(1))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);
			
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

			
			dnnl::primitive_attr attr;
			attr.set_scales_mask(DNNL_ARG_SRC_0, 0);
			attr.set_scales_mask(DNNL_ARG_SRC_1, 0);
			
			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc, attr));
			
			//fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));
			
			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			
            bwdDescFirst = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputsBwd[first]->DiffDstMemDesc, *DiffDstMemDesc, *InputsBwd[first]->DiffDstMemDesc, attr));
            bwdDescSecond = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputsBwd[second]->DiffDstMemDesc, *DiffDstMemDesc, *InputsBwd[second]->DiffDstMemDesc, attr));		

			// const auto fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } };
			// fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
            bwdFirst = std::make_unique<dnnl::binary>(dnnl::binary(*bwdDescFirst));
            bwdSecond = std::make_unique<dnnl::binary>(dnnl::binary(*bwdDescSecond));
#endif
		}
		
/*
		void ForwardProp([[maybe_unused]] const UInt batchSize, const bool training) final override
		{
			const auto fullDepth = !training || (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			
            scales[first] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scales[second] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);
            
            scaleFirst.set_host_scalar_value(scales[first]);
            scaleSecond.set_host_scalar_value(scales[second]);
            
            fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scaleFirst }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scaleSecond } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();
			
#ifndef DNN_LEAN
			if (training)
				fast_memzero(NeuronsD1.data(), PaddedCDHW() * batchSize * sizeof(Float));
#endif // DNN_LEAN
		}

        void BackwardProp([[maybe_unused]] const UInt batchSize) final override
        {
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto fullDepth = (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			
            scales[first] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scales[second] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);

            scaleFirst.set_host_scalar_value(Float(1));
			scaleSecond.set_host_scalar_value(scales[first]);

            const auto diffDestMem = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());
            
            auto firstDiffSrcMem = dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data());

            bwdArgsFirst = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, firstDiffSrcMem }, { DNNL_ARG_SRC_1, diffDestMem }, { DNNL_ARG_DST, firstDiffSrcMem }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scaleFirst }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scaleSecond } };

            #ifdef DNN_CACHE_PRIMITIVES
				bwdFirst->execute(Device.stream, bwdArgsFirst);
            #else
                dnnl::binary(*bwdDescFirst).execute(Device.stream, bwdArgsFirst);
            #endif
			Device.stream.wait();

            scaleFirst.set_host_scalar_value(Float(1));
			scaleSecond.set_host_scalar_value(scales[second]);

            auto secondDiffSrcMem = dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data());

            bwdArgsSecond = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, secondDiffSrcMem }, { DNNL_ARG_SRC_1, diffDestMem }, { DNNL_ARG_DST, secondDiffSrcMem }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scaleFirst }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scaleSecond } };
		
            #ifdef DNN_CACHE_PRIMITIVES
				bwdSecond->execute(Device.stream, bwdArgsSecond);
            #else
                dnnl::binary(*bwdDescSecond).execute(Device.stream, bwdArgsSecond);
            #endif
			Device.stream.wait();

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
        }

*/
		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto fullDepth = !training || (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			
            scales[first] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scales[second] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);
            
            scaleFirst.set_host_scalar_value(scales[first]);
            scaleSecond.set_host_scalar_value(scales[second]);

			if (training)
			{
				if constexpr (Reference || ReferenceAdd)
				{
					fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scaleFirst }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scaleSecond } };

#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();

#ifndef DNN_LEAN
					fast_memzero(NeuronsD1.data(), PaddedCDHW() * batchSize * sizeof(Float));
#endif // DNN_LEAN
				}
				else
				{
					const auto plain = IsPlainFormat();
					const auto size = GetElementsCount();
					const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * size, FwdTrainingWeight);
					const auto strideHW = HW() * VectorSize;
					
					if (plain)
					{
						if (EqualChannels(Inputs))
						{
							if (EqualDimensions(Inputs))
							{
								if (fullDepth)
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											PRAGMA_OMP_SIMD()
											for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
											{
												Neurons[hw] = Inputs[first]->Neurons[hw] + Inputs[second]->Neurons[hw];
#ifndef DNN_LEAN
												NeuronsD1[hw] = 0;
#endif	
											}
										}
									});
								else
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											PRAGMA_OMP_SIMD()
											for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
											{
												Neurons[hw] = (Inputs[first]->Neurons[hw] * scales[first]) + (Inputs[second]->Neurons[hw] * scales[second]);
#ifndef DNN_LEAN
												NeuronsD1[hw] = 0;
#endif	
											}
										}
									});
							}
							else
							{
								if (fullDepth)
									for_i(batchSize, threads,[=](UInt n)
									{
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, c, 0, 0);
											PRAGMA_OMP_SIMD()
											for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
											{
												Neurons[hw] = Inputs[first]->Neurons[hw] + Inputs[second]->Neurons[channelOffset];
#ifndef DNN_LEAN
												NeuronsD1[hw] = 0;
#endif
											}
										}
									});
								else
									for_i(batchSize, threads,[=](UInt n)
									{
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, c, 0, 0);
											PRAGMA_OMP_SIMD()
											for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
											{
												Neurons[hw] = (Inputs[first]->Neurons[hw] * scales[first]) + (Inputs[second]->Neurons[channelOffset] * scales[second]);
#ifndef DNN_LEAN
												NeuronsD1[hw] = 0;
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
								if (fullDepth)
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = Inputs[second]->OffsetPlainMem(n, 0, 0, 0);
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											for (auto hw = 0ull; hw < HW(); hw++)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) + VecFloat().load_a(&Inputs[second]->Neurons[hw + channelOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								else
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = Inputs[second]->OffsetPlainMem(n, 0, 0, 0);
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
											for (auto hw = 0ull; hw < HW(); hw++)
											{
												((VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * scales[first]) + (VecFloat().load_a(&Inputs[second]->Neurons[hw + channelOffset]) * scales[second])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
							}
						}
					}
					else
					{
						if (EqualChannels(Inputs))
						{
							if (EqualDimensions(Inputs)) // same H and W
							{
								if (fullDepth)
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw]) + VecFloat().load_a(&Inputs[second]->Neurons[hw])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw]);
#endif
											}
										}
									});
								else
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
											{
												((VecFloat().load_a(&Inputs[first]->Neurons[hw]) * scales[first]) + (VecFloat().load_a(&Inputs[second]->Neurons[hw]) * scales[second])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw]);
#endif
											}
										}
									});
							}
							else
							{
								if (fullDepth)
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw]) + VecFloat().load_a(&Inputs[second]->Neurons[channelOffset])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw]);
#endif
											}
										}
									});
								else
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
											{
												((VecFloat().load_a(&Inputs[first]->Neurons[hw]) * scales[first]) + (VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]) * scales[second])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw]);
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
								if (fullDepth)
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, 0, 0, 0);
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) + Inputs[second]->Neurons[hw + channelOffset]).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								else
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, 0, 0, 0);
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
											for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											{
												((VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * scales[first]) + (VecFloat().load_a(&Inputs[second]->Neurons[hw + channelOffset]) * scales[second])).store_a(&Neurons[hw + outputOffset]);
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
			}
			else
			{
				fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scaleFirst }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scaleSecond } };
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
			const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * size, BwdTrainingWeight);
			const auto strideHW = HW() * VectorSize;
			const auto fullDepth = (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			
			scales[first] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scales[second] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);
			const VecFloat scaleFirstVec(scales[first]);
			const VecFloat scaleSecondVec(scales[second]);

			if (EqualChannels(Inputs))
			{
				if (EqualDimensions(Inputs))
				{
					if (plain)
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw];
										InputsBwd[second]->NeuronsD1[hw] += NeuronsD1[hw];
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw] * scales[first];
										InputsBwd[second]->NeuronsD1[hw] += NeuronsD1[hw] * scales[second];
									}
								}
							});
					}
					else
					{	
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[hw])).store_a(&InputsBwd[second]->NeuronsD1[hw]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										mul_add(neuronsD1, scaleFirstVec, VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										mul_add(neuronsD1, scaleSecondVec, VecFloat().load_a(&InputsBwd[second]->NeuronsD1[hw])).store_a(&InputsBwd[second]->NeuronsD1[hw]);
									}
								}
							});
					}
				}
				else
				{
					if (plain)
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw];
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw];
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw] * scales[first];
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw] * scales[second];
									}
								}
							});
					}
					else
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									const auto channelOffset = InputsBwd[second]->OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									const auto channelOffset = InputsBwd[second]->OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										mul_add(neuronsD1, scaleFirstVec, VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										mul_add(neuronsD1, scaleSecondVec, VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
					}
				}
			}
			else
			{
				if (EqualDimensions(Inputs))
				{
					if (!plain)
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto channelOffset = InputsBwd[second]->OffsetPaddedMem(n, 0, 0, 0);
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
										(VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]) + neuronsD1).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
										(VecFloat().load_a(&InputsBwd[second]->NeuronsD1[hw + channelOffset]) + neuronsD1).store_a(&InputsBwd[second]->NeuronsD1[hw + channelOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto channelOffset = InputsBwd[second]->OffsetPaddedMem(n, 0, 0, 0);
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
										mul_add(neuronsD1, scaleFirstVec, VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										mul_add(neuronsD1, scaleSecondVec, VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
					}
					else
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, 0, 0, 0);
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset];
										InputsBwd[second]->NeuronsD1[hw + channelOffset] += NeuronsD1[hw + outputOffset];
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto channelOffset = InputsBwd[second]->OffsetPlainMem(n, 0, 0, 0);
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = OffsetPlainMem(n, c, 0, 0);
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * scales[first];
										InputsBwd[second]->NeuronsD1[hw + channelOffset] += NeuronsD1[hw + outputOffset] * scales[second];
									}
								}
							});
					}
				}
			}
#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}
