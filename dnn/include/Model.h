#pragma once
#include "Activation.h"
#include "Add.h"
#include "Average.h"
#include "AvgPooling.h"
#include "BatchNorm.h"
#include "BatchNormActivation.h"
#include "BatchNormActivationDropout.h"
#include "BatchNormRelu.h"
#include "ChannelSplit.h"
#include "ChannelSplitRatioLeft.h"
#include "ChannelSplitRatioRight.h"
#include "ChannelZeroPad.h"
#include "Concat.h"
#include "Convolution.h"
#include "ConvolutionTranspose.h"
#include "Cost.h"
#include "Dense.h"
#include "DepthwiseConvolution.h"
#include "Divide.h"
#include "Dropout.h"
#include "GlobalAvgPooling.h"
#include "GlobalMaxPooling.h"
#include "GroupNorm.h"
#include "Input.h"
#include "LayerNorm.h"
#include "LocalResponseNorm.h"
#include "LogSoftmax.h"
#include "Max.h"
#include "MaxPooling.h"
#include "Min.h"
#include "Multiply.h"
#include "PartialDepthwiseConvolution.h"
#include "PRelu.h"
#include "Reduction.h"
#include "Resampling.h"
#include "Shuffle.h"
#include "Softmax.h"
#include "Substract.h"

#include "CsvFile.h"

namespace dnn
{
	enum class TaskStates
	{
		Paused = 0,
		Running = 1,
		Stopped = 2
	};

	enum class States
	{
		Idle = 0,
		NewEpoch = 1,
		Testing = 2,
		Training = 3,
		SaveWeights = 4,
		Completed = 5
	};

	struct CheckMsg
	{
		UInt Row;
		UInt Column;
		bool Error;
		char Message[512];
				
		CheckMsg(const UInt row = 0, const UInt column = 0, const std::string& message = "", const bool error = true) :
			Row(row),
			Column(column),
			Error(error)
		{
			auto i = 0ull;
			for (auto token : message)
				Message[i++] = token;
			Message[i] = '\0';
		}
	};
	
	struct TrainingStrategy
	{
		Float Epochs;
		UInt N;
		UInt D;
		UInt H;
		UInt W;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		Float Momentum;
		Float Beta2;
		Float Gamma;
		Float L2Penalty;
		Float Dropout;
		bool HorizontalFlip;
		bool VerticalFlip;
		Float InputDropout;
		Float Cutout;
		bool CutMix;
		Float AutoAugment;
		Float ColorCast;
		UInt ColorAngle;
		Float Distortion;
		Interpolations Interpolation;
		Float Scaling;
		Float Rotation;

		bool operator==(const TrainingStrategy& o) const
		{
			return 
				std::tie(Epochs, N, D, H, W, PadD, PadH, PadW, Momentum, Beta2, Gamma, L2Penalty, Dropout, HorizontalFlip, VerticalFlip, InputDropout, Cutout, CutMix, AutoAugment, ColorCast, ColorAngle, Distortion, Interpolation, Scaling, Rotation) 
				== 
				std::tie(o.Epochs, o.N, o.D, o.H, o.W, o.PadD, o.PadH, o.PadW, o.Momentum, o.Beta2, o.Gamma, o.L2Penalty, o.Dropout, o.HorizontalFlip, o.VerticalFlip, o.InputDropout, o.Cutout, o.CutMix, o.AutoAugment, o.ColorCast, o.ColorAngle, o.Distortion, o.Interpolation, o.Scaling, o.Rotation);
		}

		TrainingStrategy() :
			Epochs(Float(1)),
			N(128ull),
			D(1ull),
			H(32ull),
			W(32ull),
			PadD(0ull),
			PadH(4ull),
			PadW(4ull),
			Momentum(Float(0.9)),
			Beta2(Float(0.999)),
			Gamma(Float(0.9)),
			L2Penalty(Float(0.0005)),
			Dropout(Float(0)),
			HorizontalFlip(true),
			VerticalFlip(false),
			InputDropout(Float(0)),
			Cutout(Float(0.7)),
			CutMix(true),
			AutoAugment(Float(0.7)),
			ColorCast(Float(0.7)),
			ColorAngle(16ull),
			Distortion(Float(0.7)),
			Interpolation(Interpolations::Cubic),
			Scaling(Float(10)),
			Rotation(Float(12))
		{
		}

		TrainingStrategy(const Float epochs, const UInt n, const UInt d, const UInt h, const UInt w, const UInt padD, const UInt padH, const UInt padW, const Float momentum, const Float beta2, const Float gamma, const Float l2penalty, const Float dropout, const bool horizontalFlip, const bool verticalFlip, const Float inputDropout, const Float cutout, const bool cutMix, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float scaling, const Float rotation) :
			Epochs(epochs),
			N(n),
			D(d),
			H(h),
			W(w),
			PadD(padD),
			PadH(padH),
			PadW(padW),
			Momentum(momentum),
			Beta2(beta2),
			Gamma(gamma),
			L2Penalty(l2penalty),
			Dropout(dropout),
			HorizontalFlip(horizontalFlip),
			VerticalFlip(verticalFlip),
			InputDropout(inputDropout),
			Cutout(cutout),
			CutMix(cutMix),
			AutoAugment(autoAugment),
			ColorCast(colorCast),
			ColorAngle(colorAngle),
			Distortion(distortion),
			Interpolation(interpolation),
			Scaling(scaling),
			Rotation(rotation)
		{
			if (Epochs <= 0 || Epochs > 1)
				throw std::invalid_argument("Epochs out of range in TrainingStrategy");
			if (N == 0)
				throw std::invalid_argument("N cannot be zero in TrainingStrategy");
			if (D == 0)
				throw std::invalid_argument("D cannot be zero in TrainingStrategy");
			if (H == 0)
				throw std::invalid_argument("H cannot be zero in TrainingStrategy");
			if (W == 0)
				throw std::invalid_argument("W cannot be zero in TrainingStrategy");
			if (PadD == 0)
				throw std::invalid_argument("PadD cannot be zero in TrainingStrategy");
			if (PadH == 0)
				throw std::invalid_argument("PadH cannot be zero in TrainingStrategy");
			if (PadW == 0)
				throw std::invalid_argument("PadW cannot be zero in TrainingStrategy");
			if (Momentum < 0 || Momentum > 1)
				throw std::invalid_argument("Momentum out of range in TrainingStrategy");
			if (Beta2 < 0 || Beta2 > 1)
				throw std::invalid_argument("Beta2 out of range in TrainingStrategy");
			if (Gamma < 0 || Gamma > 1)
				throw std::invalid_argument("Gamma out of range in TrainingStrategy");
			if (L2Penalty < 0 || L2Penalty > 1)
				throw std::invalid_argument("L2Penalty out of range in TrainingStrategy");
			if (Dropout < 0 || Dropout >= 1)
				throw std::invalid_argument("Dropout out of range in TrainingStrategy");
			if (InputDropout < 0 || InputDropout >= 1)
				throw std::invalid_argument("InputDropout out of range in TrainingStrategy");
			if (Cutout < 0 || Cutout > 1)
				throw std::invalid_argument("Cutout out of range in TrainingStrategy");
			if (AutoAugment < 0 || AutoAugment > 1)
				throw std::invalid_argument("AutoAugment out of range in TrainingStrategy");
			if (ColorCast < 0 || ColorCast > 1)
				throw std::invalid_argument("ColorCast out of range in TrainingStrategy");
			if (ColorAngle > 180)
				throw std::invalid_argument("ColorAngle cannot be zero in TrainingStrategy");
			if (Distortion < 0 || Distortion > 1)
				throw std::invalid_argument("Distortion out of range in TrainingStrategy");
			if (Scaling < 0 || Scaling > 100)
				throw std::invalid_argument("Scaling out of range in TrainingStrategy");
			if (Rotation < 0 || Rotation > 180)
				throw std::invalid_argument("Rotation out of range in TrainingStrategy");
		}
	};

	struct TrainingInfo
	{
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt Cycle;
		UInt Epoch;
		UInt SampleIndex;
		Float Rate;
		Optimizers Optimizer;
		Float Momentum;
		Float Beta2;
		Float Gamma;
		Float L2Penalty;
		Float Dropout;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		UInt PadH;
		UInt PadW;
		bool HorizontalFlip;
		bool VerticalFlip;
		Float InputDropout;
		Float Cutout;
		bool CutMix;
		Float AutoAugment;
		Float ColorCast;
		UInt ColorAngle;
		Float Distortion;
		Interpolations Interpolation;
		Float Scaling;
		Float Rotation;
		Float AvgTrainLoss;
		Float TrainErrorPercentage;
		UInt TrainErrors;
		Float AvgTestLoss;
		Float TestErrorPercentage;
		UInt TestErrors;
		Float SampleSpeed;
		States State;
		TaskStates TaskState;
	};

	struct TestingInfo
	{
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt Cycle;
		UInt Epoch;
		UInt SampleIndex;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		UInt PadH;
		UInt PadW;
		Float AvgTestLoss;
		Float TestErrorPercentage;
		UInt TestErrors;
		Float SampleSpeed;
		States State;
		TaskStates TaskState;
	};

	struct LayerInfo
	{
		char Name[512];
		char Description[2048];
		//std::string Name;
		//std::string Description;
		LayerTypes LayerType;
		Activations Activation;
		Algorithms Algorithm;
		ReduceOperations ReduceOperation;
		Costs Cost;
		UInt NeuronCount;
		UInt WeightCount;
		UInt BiasesCount;
		UInt LayerIndex;
		UInt InputsCount;
		UInt C;
		UInt D;
		UInt H;
		UInt W;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		UInt KernelH;
		UInt KernelW;
		UInt StrideH;
		UInt StrideW;
		UInt DilationH;
		UInt DilationW;
		UInt Multiplier;
		UInt Groups;
		UInt Group;
		UInt LocalSize;
		Float Dropout;
		Float LabelTrue;
		Float LabelFalse;
		Float Weight;
		UInt GroupIndex;
		UInt LabelIndex;
		UInt InputC;
		Float Alpha;
		Float Beta;
		Float K;
		Float fH;
		Float fW;
		Float P;
		Float Eps;
		bool HasBias;
		bool Scaling;
		bool AcrossChannels;
		bool Locked;
		bool Lockable;
	};

	struct ModelInfo
	{
		char Name[512];
		Datasets Dataset;
		Costs CostFunction;
		UInt LayerCount;
		UInt CostLayerCount;
		UInt CostIndex;
		UInt GroupIndex;
		UInt LabelIndex;
		UInt Hierarchies;
		UInt TrainSamplesCount;
		UInt TestSamplesCount;
		bool MeanStdNormalization;
		Float MeanTrainSet[3];
		Float StdTrainSet[3];
	};

	struct CostInfo
	{
		UInt TrainErrors;
		Float TrainLoss;
		Float AvgTrainLoss;
		Float TrainErrorPercentage;
		
		UInt TestErrors;
		Float TestLoss;
		Float AvgTestLoss;
		Float TestErrorPercentage;
	};

	struct StatsInfo
	{
		Stats NeuronsStats;
		Stats WeightsStats;
		Stats BiasesStats;
		Float FPropLayerTime;
		Float BPropLayerTime;
		Float UpdateLayerTime;
		Float FPropTime;
		Float BPropTime;
		Float UpdateTime;
		bool Locked;
		char Description[2048];
	};

	struct LogRecord
	{
		UInt Cycle;
		UInt Epoch;
		UInt GroupIndex;
		UInt CostIndex;
		std::string CostName;
		// Resolution
		UInt N;
		UInt D;
		UInt H;
		UInt W;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		// Regularization
		Optimizers Optimizer;
		Float Rate;
		Float Eps;
		Float Momentum;
		Float Beta2;
		Float Gamma;
		Float L2Penalty;
		Float Dropout;
		// Augmentation
		Float InputDropout;
		Float Cutout;
		bool CutMix;
		Float AutoAugment;
		bool HorizontalFlip;
		bool VerticalFlip;
		Float ColorCast;
		UInt ColorAngle;
		Float Distortion;
		Interpolations Interpolation;
		Float Scaling;
		Float Rotation;
		// Train
		Float AvgTrainLoss;
		UInt TrainErrors;
		Float TrainErrorPercentage;
		Float TrainAccuracy;
		// Test
		Float AvgTestLoss;
		UInt TestErrors;
		Float TestErrorPercentage;
		Float TestAccuracy;
		// Duration
		long long ElapsedMilliSeconds;
		std::string ElapsedTime;
	};

	struct Flip
	{
		bool Horizontal;
		bool Vertical;

		Flip() :
			Horizontal(false),
			Vertical(false)
		{
		}

		Flip(const bool horizontal, const bool vertical) :
			Horizontal(horizontal),
			Vertical(vertical)
		{
		}

		void ToggleFlipHorizontal()
		{
			Horizontal = !Horizontal;
		}

		void ToggleFlipVertical()
		{
			Vertical = !Vertical;
		}
	};

	static bool IsBatchNorm(const LayerTypes& type)
	{
		return std::string(magic_enum::enum_name<LayerTypes>(type)).find("BatchNorm", 0) != std::string::npos;
	}

	
	class Model
	{
	private:
		std::future<void> Task;

	public:
		std::string Name;
		std::string Script;
		std::string Definition;
		const dnnl::engine Engine;
		dnn::Device Device;
		dnnl::memory::format_tag Format;
		Dataprovider* DataProv;
		Datasets Dataset;
		std::atomic<States> State;
		std::atomic<TaskStates> TaskState;
		Costs CostFunc;
		Optimizers Optimizer;
		UInt CostIndex;
		UInt LabelIndex;
		UInt GroupIndex;
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt CurrentCycle;
		UInt CurrentEpoch;
		UInt SampleIndex;
		//UInt LogInterval;
		UInt GotoEpoch;
		UInt GotoCycle;
		UInt AdjustedTrainSamplesCount;
		UInt AdjustedTestSamplesCount;
		UInt TrainSkipCount;
		UInt TestSkipCount;
		UInt TrainOverflowCount;
		UInt TestOverflowCount;
		UInt N;
		UInt C;
		UInt D;
		UInt H;
		UInt W;
		UInt PadC;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		bool MirrorPad;
		bool RandomCrop;
		bool MeanStdNormalization;
		bool FixedDepthDrop;
		Float DepthDrop;
		Float Ratio;
		Fillers WeightsFiller;
		FillerModes WeightsFillerMode;
		Float WeightsGain;
		Float WeightsScale;
		Float WeightsLRM;
		Float WeightsWDM;
		Fillers BiasesFiller;
		FillerModes BiasesFillerMode;
		Float BiasesGain;
		Float BiasesScale;
		Float BiasesLRM;
		Float BiasesWDM;
		Float AlphaFiller;
		Float BetaFiller;
		Float BatchNormMomentum;
		Float BatchNormEps;
		Float Dropout;
		//CostInfo CostInfo;
		UInt TrainErrors;
		UInt TestErrors;
		Float TrainLoss;
		Float TestLoss;
		Float AvgTrainLoss;
		Float AvgTestLoss;
		Float TrainErrorPercentage;
		Float TestErrorPercentage;
		Float TrainAccuracy;
		Float TestAccuracy;
		Float SampleSpeed;
		Float Rate;
		bool BatchNormScaling;
		bool HasBias;
		bool PersistOptimizer;
		bool DisableLocking;
		std::vector<Flip> TrainSamplesFlip;
		std::vector<Flip> TestSamplesFlip;
		std::vector<UInt> RandomTrainSamples;
		TrainingRate CurrentTrainingRate;
		std::vector<TrainingRate> TrainingRates;
		std::vector<TrainingStrategy> TrainingStrategies;
		bool UseTrainingStrategy;
		std::vector<LogRecord> TrainingLog;
		std::vector<std::unique_ptr<Layer>> Layers;
		std::vector<Cost*> CostLayers;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::atomic<UInt> FirstUnlockedLayer;
		std::atomic<bool> BatchSizeChanging;
		std::atomic<bool> ResettingWeights;
		
		void(*NewEpoch)(UInt, UInt, UInt, UInt, Float, Float, Float, bool, bool, Float, Float, bool, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, UInt, UInt, UInt, UInt, UInt, UInt, Float, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt, UInt);

		auto GetModelName(const std::string& definition) const
		{
			if (definition.length() > 2)
			{
				auto first = definition.find_first_of("]");
				if (first != std::string::npos)
					return definition.substr(1, first - 1);
			}

			return std::string("");
		}

		auto GetDatasetEnum(const std::string& definition) const
		{
			if (definition.length() > 2)
			{
				auto start = definition.find("Dataset=");
				if (start != std::string::npos)
				{
					auto end = definition.find(nwl, start);
					auto datasetStr = definition.substr(start + 8ull, end - 1ull);
					return magic_enum::enum_cast<Datasets>(datasetStr).value_or(Datasets::cifar10);
				}
			}

			return Datasets::cifar10;
		}

		auto GetLogFileName(const std::string& modelName, const Datasets dataset) const
		{
			return modelName + std::string("-(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(dataset))) + std::string(").csv");
		}

		auto GetWeightsFileName(const bool persistOptimizer, const Datasets dataset, const Optimizers optimizer) const
		{
			auto fileName = std::string("(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(dataset)));

			if (persistOptimizer)
				fileName += std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(optimizer)));
			
			return fileName + std::string(").bin");
		}

		Model(const std::string& definition, Dataprovider* dataprovider) :
			Name(GetModelName(definition)),
			Definition(definition),
			Engine(dnnl::engine(dnnl::engine::kind::cpu, 0)),
			Device(dnn::Device(Engine, dnnl::stream(Engine))),
			Format(dnnl::memory::format_tag::any),
			DataProv(dataprovider),
			Dataset(GetDatasetEnum(definition)),	// Dataset
			State(States::Idle),
			TaskState(TaskStates::Stopped),
			CostFunc(Costs::CategoricalCrossEntropy),
			Optimizer(Optimizers::SGD),
			CostIndex(0),
			LabelIndex(0),
			GroupIndex(0),
			TotalCycles(0),
			TotalEpochs(0),
			CurrentEpoch(1),
			CurrentCycle(1),
			SampleIndex(0),
			//LogInterval(10000),
			GotoEpoch(1),
			AdjustedTrainSamplesCount(0),
			AdjustedTestSamplesCount(0),
			TrainSkipCount(0),
			TestSkipCount(0),
			TrainOverflowCount(0),
			TestOverflowCount(0),
			N(1),
			C(3),									// Dim
			D(1),
			H(32),
			W(32),
			PadD(0),
			PadC(0),
			PadH(0),
			PadW(0),
			MirrorPad(false),						// MirrorPad or ZeroPad
			RandomCrop(false),						// RandomCrop
			MeanStdNormalization(true),				// MeanStd
			FixedDepthDrop(false),					// FixedDepthDrop
			DepthDrop(Float(0.0)),					// DepthDrop			(Stochastic Depth: https://www.arxiv-vanity.com/papers/1603.09382/)
			Ratio(Float(0.375)),                    // Ratio
			WeightsFiller(Fillers::HeNormal),		// WeightsFiller
			WeightsFillerMode(FillerModes::In),		// WeightsFillerMode
			WeightsGain(Float(1)),					// WeightsGain
			WeightsScale(Float(0.05)),				// WeightsScale
			WeightsLRM(Float(1)),					// WeightsLRM
			WeightsWDM(Float(1)),					// WeightsWDM
			BiasesFiller(Fillers::Constant),		// BiasesFiller
			BiasesFillerMode(FillerModes::In),		// BiasesFillerMode
			BiasesGain(Float(1)),					// BiasesGain
			BiasesScale(Float(0)),					// BiasesScale
			BiasesLRM(Float(1)),					// BiasesLRM
			BiasesWDM(Float(1)),					// BiasesWDM
			AlphaFiller(Float(0)),					// Alpha
			BetaFiller(Float(0)),					// Beta
			BatchNormMomentum(Float(0.995)),		// Momentum
			BatchNormEps(Float(1e-04)),				// Eps
			Dropout(Float(0)),						// Dropout
			TrainErrors(0),
			TestErrors(0),
			TrainLoss(Float(0)),
			TestLoss(Float(0)),
			AvgTrainLoss(Float(0)),
			AvgTestLoss(Float(0)),
			TrainErrorPercentage(Float(0)),
			TestErrorPercentage(Float(0)),
			TrainAccuracy(Float(0)),
			TestAccuracy(Float(0)),
			SampleSpeed(Float(0)),
			Rate(Float(0)),
			BatchNormScaling(true),					// Scaling
			HasBias(true),							// Biases
			PersistOptimizer(false),
			DisableLocking(true),
			NewEpoch(nullptr),
			TrainingRates(std::vector<TrainingRate>()),
			TrainingStrategies(std::vector<TrainingStrategy>()),
			UseTrainingStrategy(false),
			TrainingLog(std::vector<LogRecord>()),
			Layers(std::vector<std::unique_ptr<Layer>>()),
			CostLayers(std::vector<Cost*>()),
			fpropTime(std::chrono::duration<Float>(Float(0))),
			bpropTime(std::chrono::duration<Float>(Float(0))),
			updateTime(std::chrono::duration<Float>(Float(0))),
			FirstUnlockedLayer(1),
			BatchSizeChanging(false),
			ResettingWeights(false)
		{
#ifdef DNN_LOG
			dnnl_set_verbose(2);
#else
			dnnl_set_verbose(0);
#endif

#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx512_core_bf16);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_AVX2)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx2);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_AVX)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::sse41);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::no_hints);
#endif
			//dnnl::set_primitive_cache_capacity(1000);
			//dnnl::set_default_fpmath_mode(dnnl::fpmath_mode::any);

			if (DataProv != nullptr)
				LoadLog((DataProv->StorageDirectory / std::string("state") / GetLogFileName(Name, Dataset)).string());
		}

		virtual ~Model() = default;
		
		auto GetWeightsSize(const bool persistOptimizer, const Optimizers optimizer) const
		{
			auto weightsSize = std::streamsize(0);

			for (const auto& layer : Layers)
				weightsSize += layer->GetWeightsSize(persistOptimizer, optimizer);

			return weightsSize;
		}

		auto GetNeuronsSize(const UInt batchSize) const
		{
			auto neuronsSize = UInt(0);

			for (const auto& layer : Layers)
				neuronsSize += layer->GetNeuronsSize(batchSize);

			return neuronsSize;
		}

		bool BatchNormUsed() const
		{
			for (const auto& layer : Layers)
				if (IsBatchNorm(layer->LayerType))
					return true;

			return false;
		}

		bool ChangeResolution(const UInt n, const UInt d, const UInt h, const UInt w, const UInt padD, const UInt padH, const UInt padW)
		{
			if (n < 1 || d < 1 || h < 1 || w < 1)
				return false;
			
			if ((n == N) && (d == D) && (h == H) && (w == W))
			{
				PadD = padD;
				PadH = padH;
				PadW = padW;

				AdjustedTrainSamplesCount = (DataProv->TrainSamplesCount % N == 0) ? DataProv->TrainSamplesCount : ((DataProv->TrainSamplesCount / N) + 1) * N;
				AdjustedTestSamplesCount = (DataProv->TestSamplesCount % N == 0) ? DataProv->TestSamplesCount : ((DataProv->TestSamplesCount / N) + 1) * N;
				TrainSkipCount = N - (AdjustedTrainSamplesCount - DataProv->TrainSamplesCount);
				TestSkipCount = N - (AdjustedTestSamplesCount - DataProv->TestSamplesCount);
				TrainOverflowCount = AdjustedTrainSamplesCount - N;
				TestOverflowCount = AdjustedTestSamplesCount - N;

				return true;
			}
						
			const auto currentSize = GetNeuronsSize(N) + GetWeightsSize(PersistOptimizer, Optimizer);

			while (BatchSizeChanging.load() || ResettingWeights.load())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				std::this_thread::yield();
			}

			BatchSizeChanging.store(true);

			if (Layers[0]->D != d || Layers[0]->H != h || Layers[0]->W != w)
			{
				Layers[0]->D = d;
				Layers[0]->H = h;
				Layers[0]->W = w;
				for (auto& layer : Layers)
					layer->UpdateResolution();
			}

			if ((n * d * h * w) > (N * D * H * W))
			{
				auto requestedSize = GetNeuronsSize(n) + GetWeightsSize(PersistOptimizer, Optimizer);

				if (GetTotalFreeMemory() + currentSize < requestedSize)
				{
					std::cout << std::string("Memory required: ") << std::to_string(requestedSize / 1024 / 1024) << std::string(" MB with resolution") << std::to_string(n) + std::string("x") + std::to_string(h) + std::string("x") + std::to_string(w) << std::endl << std::endl;

					Layers[0]->D = D;
					Layers[0]->H = H;
					Layers[0]->W = W;
					for (auto& layer : Layers)
						layer->UpdateResolution();
					
					BatchSizeChanging.store(false);

					return false;
				}
			}

			for (auto& layer : Layers)
				layer->SetBatchSize(n);
			

			N = n;
			D = d;
			H = h;
			W = w;
			PadD = padD;
			PadH = padH;
			PadW = padW;

			AdjustedTrainSamplesCount = (DataProv->TrainSamplesCount % N == 0) ? DataProv->TrainSamplesCount : ((DataProv->TrainSamplesCount / N) + 1) * N;
			AdjustedTestSamplesCount = (DataProv->TestSamplesCount % N == 0) ? DataProv->TestSamplesCount : ((DataProv->TestSamplesCount / N) + 1) * N;
			TrainSkipCount = N - (AdjustedTrainSamplesCount - DataProv->TrainSamplesCount);
			TestSkipCount = N - (AdjustedTestSamplesCount - DataProv->TestSamplesCount);
			TrainOverflowCount = AdjustedTrainSamplesCount - N;
			TestOverflowCount = AdjustedTestSamplesCount - N;
			
			BatchSizeChanging.store(false);

			return true;
		}

		void ChangeDropout(const Float dropout, const UInt batchSize)
		{
			if (dropout < 0 || dropout >= 1)
				throw std::invalid_argument("Invalid dropout value in ChangeDropout function");

			for (auto& layer : Layers)
			{
				switch (layer->LayerType)
				{
				case LayerTypes::BatchNormActivationDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				
				case LayerTypes::Dropout:
				{
					auto drop = dynamic_cast<dnn::Dropout*>(layer.get());
					if (drop)
					{
						drop->UpdateDropout(dropout);
						drop->SetBatchSize(batchSize);
					}
				}
				break;

				default:
					break;
				}
			}

			Dropout = dropout;
		}

		bool SetFormat(bool plain = false)
		{
			if (TaskState.load() == TaskStates::Stopped)
			{
				Format = plain ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::any;
				for (auto& layer : Layers)
					layer->NeuronsFormat = Format;
				
				return true;
			}
			else
			    return false;
		}

		void ResetWeights()
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				ResettingWeights.store(true);

				for (auto& layer : Layers)
				{
					while (layer->RefreshingStats.load())
						std::this_thread::sleep_for(std::chrono::milliseconds(100));

					layer->ResetWeights(WeightsFiller, WeightsFillerMode, WeightsGain, WeightsScale, BiasesFiller, BiasesFillerMode, BiasesGain, BiasesScale);
					layer->ResetOptimizer(Optimizer);
				}

				ResettingWeights.store(false);
			}
		}

		bool IsUniqueLayerName(std::string name) const
		{
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			
			for (const auto& layer : Layers)
			{
				auto layerName = std::string(layer->Name);
				std::transform(layerName.begin(), layerName.end(), layerName.begin(), ::tolower);
				if (layerName == name)
					return false;
			}

			return true;
		}

		void SetLocking(const bool locked)
		{
			for (auto &layer : Layers)
				if (layer->Lockable() && !DisableLocking)
					layer->LockUpdate.store(locked);
			
			if (!DisableLocking)
			{
				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
				{
					Layers[i]->bpropTime = std::chrono::duration<Float>(0);
					Layers[i]->updateTime = std::chrono::duration<Float>(0);

					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}
				}
			}
		}

		void SetLayerLocking(const UInt layerIndex, const bool locked)
		{
			if (layerIndex < Layers.size() && Layers[layerIndex]->Lockable() && !DisableLocking)
			{
				Layers[layerIndex]->LockUpdate.store(locked);

				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
				{
					Layers[i]->bpropTime = std::chrono::duration<Float>(0);
					Layers[i]->updateTime = std::chrono::duration<Float>(0);

					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}
				}
			}
		}		
	
		void AddTrainingRate(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples)
		{
			if (clear)
				TrainingRates = std::vector<TrainingRate>();

			TotalCycles = 1;
			GotoEpoch = gotoEpoch;
			
			const auto LR = rate.MaximumRate;
			const auto Epochs = rate.Epochs;

			auto decayAfterEpochs = rate.DecayAfterEpochs;
			if (rate.Epochs < decayAfterEpochs)
				decayAfterEpochs = rate.Epochs;

			auto totIteration = rate.Epochs / decayAfterEpochs;
			auto newRate = rate.MaximumRate;

			for (auto i = 0ull; i < totIteration; i++)
			{
				if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
				{
					const auto weightDecayMultiplier = newRate / LR;
					const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.N) / (Float(trainSamples) / rate.N) * Epochs, Float(0.5));

					if ((i + 1) >= gotoEpoch)
					{
						if (UseTrainingStrategy && TrainingStrategies.size() > 0)
						{
							TrainingRate tmpRate;

							auto sum = Float(0);
							for (const auto& strategy : TrainingStrategies)
							{
								tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

								sum += strategy.Epochs * rate.Epochs;
								if ((i + 1) <= sum)
									break;
							}

							TrainingRates.push_back(tmpRate);
						}
						else
							TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
					}
				}
				else
				{
					if ((i + 1) >= gotoEpoch)
					{
						if (UseTrainingStrategy && TrainingStrategies.size() > 0)
						{
							TrainingRate tmpRate;

							auto sum = Float(0);
							for (const auto& strategy : TrainingStrategies)
							{
								tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

								sum += strategy.Epochs * rate.Epochs;
								if ((i + 1) <= sum)
									break;
							}

							TrainingRates.push_back(tmpRate);
						}
						else
							TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
					}
				}

				if (newRate * rate.DecayFactor > rate.MinimumRate)
					newRate *= rate.DecayFactor;
				else
					newRate = rate.MinimumRate;
			}

			if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
			{
				const auto weightDecayMultiplier = newRate / LR;
				const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.N) / (Float(trainSamples) / rate.N) * Epochs, Float(0.5));

				if ((totIteration * decayAfterEpochs) < rate.Epochs)
				{
					if (UseTrainingStrategy && TrainingStrategies.size() > 0)
					{
						TrainingRate tmpRate;

						auto sum = Float(0);
						for (const auto& strategy : TrainingStrategies)
						{
							tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

							sum += strategy.Epochs * rate.Epochs;
							if (totIteration * decayAfterEpochs <= sum)
								break;
						}

						TrainingRates.push_back(tmpRate);
					}
					else
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}
			}
			else
			{
				if ((totIteration * decayAfterEpochs) < rate.Epochs)
				{
					if (UseTrainingStrategy && TrainingStrategies.size() > 0)
					{
						TrainingRate tmpRate;

						auto sum = Float(0);
						for (const auto& strategy : TrainingStrategies)
						{
							tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs),1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

							sum += strategy.Epochs * rate.Epochs;
							if (totIteration * decayAfterEpochs <= sum)
								break;
						}

						TrainingRates.push_back(tmpRate);
					}
					else
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}
			}
		}

		void AddTrainingRateSGDR(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt gotoCycle, const UInt trainSamples)
		{
			if (clear)
				TrainingRates = std::vector<TrainingRate>();

			TotalCycles = rate.Cycles;
			GotoEpoch = gotoEpoch;
			GotoCycle = gotoCycle;

			const auto LR = rate.MaximumRate;
			auto maxRate = rate.MaximumRate;
			auto minRate = rate.MinimumRate;
			auto epoch = 0ull;

			for (auto c = 0ull; c < TotalCycles; c++)
			{	
				if (c >= (gotoCycle-1))
				{
					const auto totalEpochs = rate.Epochs * (c > 0 ? (rate.EpochMultiplier != 1 ? c * rate.EpochMultiplier : 1) : 1);
					for (auto i = 0ull; i < totalEpochs; i++)
					{
						const auto newRate = (minRate + Float(0.5) * (maxRate - minRate) * (Float(1) + std::cos(Float(i) / Float(totalEpochs) * Float(3.1415926535897932384626433832))));

						epoch++;

						if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
						{
							const auto weightDecayMultiplier = newRate / LR;
							const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.N) / (Float(trainSamples) / rate.N) * totalEpochs, Float(0.5));

							if (epoch >= gotoEpoch)
							{
								if (UseTrainingStrategy && TrainingStrategies.size() > 0)
								{
									TrainingRate tmpRate;

									auto sum = Float(0);
									for (const auto& strategy : TrainingStrategies)
									{
										tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

										sum += strategy.Epochs * totalEpochs;
										if (epoch <= sum)
											break;
									}

									TrainingRates.push_back(tmpRate);
								}
								else
									TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
							}
						}
						else
						{
							if (epoch >= gotoEpoch)
							{
								if (UseTrainingStrategy && TrainingStrategies.size() > 0)
								{
									TrainingRate tmpRate;

									auto sum = Float(0);
									for (const auto& strategy : TrainingStrategies)
									{
										tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

										sum += strategy.Epochs * totalEpochs;
										if (epoch <= sum)
											break;
									}

									TrainingRates.push_back(tmpRate);
								}
								else
									TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
							}
						}
					}

					if (rate.DecayFactor != Float(1))
					{
						maxRate *= rate.DecayFactor;
						minRate *= rate.DecayFactor;
					}
				}
			}
		}

		bool CheckTaskState() const
		{
			while (TaskState.load() == TaskStates::Paused) 
			{ 
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::this_thread::yield(); 
			}
			
			return TaskState.load() == TaskStates::Running;
		}

		void TrainingAsync()
		{
			Task = std::async(std::launch::async, [=] { Training(); });
			//Task = std::async(std::launch::async, [=] { CheckValid(); });
		}

		void TestingAsync()
		{
			Task = std::async(std::launch::async, [=] { Testing(); });
		}

		void StopTask()
		{
			if (TaskState.load() != TaskStates::Stopped)
			{
				TaskState.store(TaskStates::Stopped);

				if (Task.valid())
					try
				    {
					    Task.get();
				    }
				    catch (const std::future_error& e)
				    {
					    std::cout << std::string("StopTask exception: ") << std::string(e.what()) << std::endl << std::string("code: ") << e.code().message() << std::endl;
				    }

				State.store(States::Completed);
			}
		}

		void PauseTask()
		{
			if (TaskState.load() == TaskStates::Running)
				TaskState.store(TaskStates::Paused);
		}

		void ResumeTask()
		{
			if (TaskState.load() == TaskStates::Paused)
				TaskState.store(TaskStates::Running);
		}

		void SetOptimizer(const Optimizers optimizer)
		{
			if (optimizer != Optimizer)
			{
				for (auto& layer : Layers)
				{
					layer->InitializeDescriptors(N);
					layer->SetOptimizer(optimizer);
				}

				Optimizer = optimizer;
			}
		}

		void ResetOptimizer()
		{
			for (auto &layer : Layers)
				layer->ResetOptimizer(Optimizer);
		}

#ifdef DNN_STOCHASTIC
		void CostFunction(const States state)
		{
			for (auto cost : CostLayers)
			{
				auto loss = Float(0);

				for (auto i = 0ull; i < cost->C; i++)
					loss += cost->Neurons[i] * cost->Weight;

				if (state == States::Training)
					cost->TrainLoss += loss;
				else
					cost->TestLoss += loss;
			}
		}

		void Recognized(const States state, const std::vector<LabelInfo>& sampleLabel)
		{
			for (auto cost : CostLayers)
			{
				const auto inputLayer = cost->InputLayer;
				const auto labelIndex = cost->LabelIndex;

				auto hotIndex = 0ull;
				auto maxValue = std::numeric_limits<Float>::lowest();
				for (auto i = 0ull; i < cost->C; i++)
				{
					if (inputLayer->Neurons[i] > maxValue)
					{
						maxValue = inputLayer->Neurons[i];
						hotIndex = i;
					}
				}

				if (hotIndex != sampleLabel[labelIndex].LabelA)
				{
					if (state == States::Training)
						cost->TrainErrors++;
					else
						cost->TestErrors++;
				}

				if (state == States::Testing)
					cost->ConfusionMatrix[hotIndex][sampleLabel[labelIndex].LabelA]++;
			}
		}
#endif

		void CostFunctionBatch(const States state, const UInt batchSize, const bool overflow, const UInt skipCount)
		{
			for (auto cost : CostLayers)
			{
				for (auto b = 0ull; b < batchSize; b++)
				{
					if (overflow && b >= skipCount)
						return;

					const auto batchOffset = b * cost->C;
					auto loss = Float(0);

					for (auto i = 0ull; i < cost->C; i++)
						loss += cost->Neurons[batchOffset + i] * cost->Weight;

					if (state == States::Training)
						cost->TrainLoss += loss;
					else
						cost->TestLoss += loss;
				}
			}
		}

		void RecognizedBatch(const States state, const UInt batchSize, const bool overflow, const UInt skipCount, const std::vector<std::vector<LabelInfo>>& sampleLabels)
		{
			for (auto cost : CostLayers)
			{
				const auto &inputLayer = cost->InputLayer;
				const auto labelIndex = cost->LabelIndex;

				for (auto b = 0ull; b < batchSize; b++)
				{
					if (overflow && b >= skipCount)
						return;

					const auto sampleOffset = b * inputLayer->C;

					auto hotIndex = 0ull;
					auto maxValue = std::numeric_limits<Float>::lowest();
					for (auto i = 0ull; i < inputLayer->C; i++)
					{
						if (inputLayer->Neurons[i + sampleOffset] > maxValue)
						{
							maxValue = inputLayer->Neurons[i + sampleOffset];
							hotIndex = i;
						}
					}

					if (hotIndex != sampleLabels[b][labelIndex].LabelA)
					{
						if (state == States::Training)
							cost->TrainErrors++;
						else
							cost->TestErrors++;
					}

					if (state == States::Testing)
						cost->ConfusionMatrix[hotIndex][sampleLabels[b][labelIndex].LabelA]++;
				}
			}
		}
		
		/*
		void SetBatchSize(const UInt batchSize)
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				BatchSizeChanging.store(true);

				for (auto& layer : Layers)
					layer->SetBatchSize(batchSize);

				AdjustedTrainSamplesCount = (DataProv->TrainSamplesCount % batchSize == 0) ? DataProv->TrainSamplesCount : ((DataProv->TrainSamplesCount / batchSize) + 1) * batchSize;
				AdjustedTestSamplesCount = (DataProv->TestSamplesCount % batchSize == 0) ? DataProv->TestSamplesCount : ((DataProv->TestSamplesCount / batchSize) + 1) * batchSize;
				TrainSkipCount = batchSize - (AdjustedTrainSamplesCount - DataProv->TrainSamplesCount);
				TestSkipCount = batchSize - (AdjustedTestSamplesCount - DataProv->TestSamplesCount);
				TrainOverflowCount = AdjustedTrainSamplesCount - batchSize;
				TestOverflowCount = AdjustedTestSamplesCount - batchSize;;

				N = batchSize;

				BatchSizeChanging.store(false);
			}
		}
		*/

		auto IsSkippable(const Layer& layer) const
		{
			return layer.LayerType == LayerTypes::Add || layer.LayerType == LayerTypes::Average || layer.LayerType == LayerTypes::Substract; // || layer.LayerType == LayerTypes::Multiply || layer.LayerType == LayerTypes::Divide;
		}

		auto GetTotalSkipConnections() const
		{
			auto totalSkipConnections = 0ull;
		
			for (const auto& layer : Layers)
				if (IsSkippable(*layer.get()))
					for (const auto& l : layer->Outputs)
						if (IsSkippable(*l))
							totalSkipConnections++;

			return totalSkipConnections;
		}

		void StochasticDepth(const UInt totalSkipConnections, const Float dropRate = Float(0.5), const bool fixed = false)
		{
			auto isSkipConnection = false;
			auto endLayer = std::string("");
			auto survive = true;
			
			auto skipConnection = 0ull;
			for (auto& layer : Layers)
			{
				if (IsSkippable(*layer.get()))
				{
					if (isSkipConnection && layer->Name == endLayer)
					{
						auto survivalProb = Float(1);

						for (auto inputLayer = 0ull; inputLayer < layer->Inputs.size(); inputLayer++)
						{
							if (!IsSkippable(*layer->Inputs[inputLayer]))
								survivalProb = fixed ? Float(1) / (Float(1) - dropRate) : Float(1) / (Float(1) - (dropRate * Float(skipConnection) / Float(totalSkipConnections)));
							else
								survivalProb = Float(1);
							
							switch (layer->LayerType)
							{
							case LayerTypes::Add:
								dynamic_cast<Add*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Average:
								dynamic_cast<Average*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Substract:
								dynamic_cast<Substract*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							/*case LayerTypes::Multiply:
								dynamic_cast<Multiply*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Divide:
								dynamic_cast<Divide*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;*/
							default:
								break;
							}
						}
					}
					
					isSkipConnection = false;
					for (const auto& outputLayer : layer->Outputs)
					{
						if (IsSkippable(*outputLayer))
						{
							isSkipConnection = true;
							endLayer = outputLayer->Name;
							skipConnection++;
							survive = Bernoulli<bool>(fixed ? Float(1) / (Float(1) - dropRate) : (Float(1) - (dropRate * Float(skipConnection) / Float(totalSkipConnections))));
						}
					}
				}
				else if (isSkipConnection)
					layer->Skip = !survive;
			}
		}

		std::vector<Layer*> GetLayerInputs(const std::vector<std::string>& inputs) const
		{
			auto list = std::vector<Layer*>();

			for (auto& name : inputs)
			{
				auto exists = false;

				for (auto& layer : Layers)
				{
					if (layer->Name == name)
					{
						list.push_back(layer.get());
						exists = true;
					}
				}

				if (!exists)
					throw std::invalid_argument((std::string("Invalid input layer: ") + name).c_str());
			}

			return list;
		}

		std::vector<Layer*> GetLayerOutputs(const Layer* parentLayer) const
		{
			auto outputs = std::vector<Layer*>();
					
			for (auto& layer : Layers)
				if (layer->Name != parentLayer->Name)
					for (auto input : layer->Inputs)
						if (input->Name == parentLayer->Name)
							outputs.push_back(layer.get());
					
			return outputs;
		}

		std::vector<Layer*> SetRelations()
		{
			// This determines how the backprop step correctly flows
			// When SharesInput is true we have to add our diff vector instead of just copying it because there's more than one layer involved

			auto unreferencedLayers = std::vector<Layer*>();

			for (auto& layer : Layers)
				layer->SharesInput = false;
			
			for (auto& layer : Layers)
			{
				layer->Outputs = GetLayerOutputs(layer.get());

				auto outputsCount = layer->Outputs.size();

				if (outputsCount > 0)
				{
					for (auto& l : Layers)
					{
						if (l->Name == layer->Name)
							continue;

						for (auto input : l->Inputs)
						{
							if (input->Name == layer->Name)
							{
								if (input->LayerType != LayerTypes::Input)
									l->SharesInput = !l->InplaceBwd;

								outputsCount--;
								break;
							}
						}
						
						if (outputsCount <= 1)
							break;
					}
				}
				else
				{
					if (outputsCount == 0 && layer->LayerType != LayerTypes::Cost)
						unreferencedLayers.push_back(layer.get());
				}
			}

			return unreferencedLayers;
		}
	
		void Training()
		{
			if (TaskState.load() == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);

				auto msg = std::string();
				if (!Activation::CheckActivations(msg))
				{
					cimg_library::cimg::dialog("Activations Sanity Check", msg.c_str(), "OK");

					State.store(States::Completed);
					return;
				};
				
				const auto totalSkipConnections = GetTotalSkipConnections();

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				auto bpropTimeCount = std::chrono::duration<Float>(Float(0));
				auto updateTimeCount = std::chrono::duration<Float>(Float(0));
                auto elapsedTime = std::chrono::duration<Float>(Float(0));

				TotalEpochs = 0;
				for (const auto& rate : TrainingRates)
					TotalEpochs += rate.Epochs;
				TotalEpochs += GotoEpoch - 1;

				auto useCycli = false;
				for (const auto& rate : TrainingRates)
					if (rate.Cycles != 1)
						useCycli = true;

				CurrentEpoch = GotoEpoch - 1;
				CurrentTrainingRate = TrainingRates[0];
				Rate = CurrentTrainingRate.MaximumRate;
				CurrentCycle = CurrentTrainingRate.Cycles;
			
				if (!ChangeResolution(CurrentTrainingRate.N, CurrentTrainingRate.D, CurrentTrainingRate.H, CurrentTrainingRate.W, CurrentTrainingRate.PadD, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
				{
					State.store(States::Completed);
					return;
				}
				
				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, N);

				auto learningRateEpochs = CurrentTrainingRate.Epochs;
				auto learningRateIndex = 0ull;

				RandomTrainSamples = std::vector<UInt>(DataProv->TrainSamplesCount);
				for (auto i = 0ull; i < DataProv->TrainSamplesCount; i++)
					RandomTrainSamples[i] = i;

				TrainSamplesFlip = std::vector<Flip>();
				TestSamplesFlip = std::vector<Flip>();
				for (auto index = 0ull; index < DataProv->TrainSamplesCount; index++)
					TrainSamplesFlip.push_back(Flip{ Bernoulli<bool>(Float(0.5)) , Bernoulli<bool>(Float(0.5)) });
				for (auto index = 0ull; index < DataProv->TestSamplesCount; index++)
					TestSamplesFlip.push_back(Flip{ Bernoulli<bool>(Float(0.5)) , Bernoulli<bool>(Float(0.5)) });
				
				SetOptimizer(CurrentTrainingRate.Optimizer);
				for (auto& layer : Layers)
					if (layer->CheckOptimizer(Optimizer))
					{
						for (auto& l : Layers)
							l->ResetOptimizer(Optimizer);
						State.store(States::Completed);
						return;
					}
			
				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}

				while (CurrentEpoch < TotalEpochs)
				{
					if (CurrentEpoch - (GotoEpoch - 1) == learningRateEpochs)
					{
						learningRateIndex++;
						CurrentTrainingRate = TrainingRates[learningRateIndex];
						Rate = CurrentTrainingRate.MaximumRate;
						
						if (!ChangeResolution(CurrentTrainingRate.N, CurrentTrainingRate.D, CurrentTrainingRate.H, CurrentTrainingRate.W, CurrentTrainingRate.PadD, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
						{
							State.store(States::Completed);
							return;
						}
								
						if (Dropout != CurrentTrainingRate.Dropout)
							ChangeDropout(CurrentTrainingRate.Dropout, N);

						learningRateEpochs += CurrentTrainingRate.Epochs;

						if (CurrentTrainingRate.Optimizer != Optimizer)
						{
							SetOptimizer(CurrentTrainingRate.Optimizer);
							for (auto& layer : Layers)
								layer->ResetOptimizer(Optimizer);
						}
					}

					timePointGlobal = timer.now();
					CurrentEpoch++;
					CurrentCycle = CurrentTrainingRate.Cycles;

					if (CurrentTrainingRate.HorizontalFlip)
						for (auto index = 0ull; index < DataProv->TrainSamplesCount; index++)
							TrainSamplesFlip[index].ToggleFlipHorizontal();

					if (CurrentTrainingRate.VerticalFlip)
						for (auto index = 0ull; index < DataProv->TrainSamplesCount; index++)
							TrainSamplesFlip[index].ToggleFlipVertical();

					if (CheckTaskState())
					{
						State.store(States::Training);

						const auto shuffleCount = UniformInt<UInt>(DataProv->ShuffleCount / 2ull, DataProv->ShuffleCount);
						for (auto shuffle = 0ull; shuffle < shuffleCount; shuffle++)
							std::shuffle(std::begin(RandomTrainSamples), std::end(RandomTrainSamples), std::mt19937(Seed<unsigned>()));

						for (auto cost : CostLayers)
							cost->Reset();

#ifdef DNN_STOCHASTIC				
						if (N == 1)
						{
							for (SampleIndex = 0; SampleIndex < DataProv->TrainSamplesCount; SampleIndex++)
							{
								if (DepthDrop > 0)
									StochasticDepth(totalSkipConnections, DepthDrop, FixedDepthDrop);

								// Forward
								const auto timePointLocal = timer.now();
								auto SampleLabel = TrainSample(SampleIndex);
								Layers[0]->fpropTime = timer.now() - timePointLocal;

								for (auto cost : CostLayers)
									cost->SetSampleLabel(SampleLabel);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									timePoint = timer.now();
									if (!Layers[i]->Skip)
										Layers[i]->ForwardProp(1, true);
									Layers[i]->fpropTime = timer.now() - timePoint;
								}

								CostFunction(State.load());
								Recognized(State.load(), SampleLabel);
								fpropTime = timer.now() - timePointLocal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (Layers[i]->HasWeights && TaskState.load() == TaskStates::Running)
									{
										timePoint = timer.now();
										if (!Layers[i]->Skip)
										{
											Layers[i]->ResetGradients();
											Layers[i]->BackwardProp(N);
										}
										Layers[i]->bpropTime = timer.now() - timePoint;
										timePoint = timer.now();
										if (!Layers[i]->Skip)
											Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
										Layers[i]->updateTime = timer.now() - timePoint;
										updateTimeCount += Layers[i]->updateTime;
									}
									else
									{
										timePoint = timer.now();
										if (!Layers[i]->Skip && TaskState.load() == TaskStates::Running)
											Layers[i]->BackwardProp(1);
										Layers[i]->bpropTime = timer.now() - timePoint;
									}
									bpropTimeCount += Layers[i]->bpropTime;
								}
								bpropTime = bpropTimeCount;
								updateTime = updateTimeCount;

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
						}
						else
						{
#endif
							auto overflow = false;
							for (SampleIndex = 0; SampleIndex < AdjustedTrainSamplesCount; SampleIndex += N)
							{
								// Forward
								if (DepthDrop > 0)
									StochasticDepth(totalSkipConnections, DepthDrop, FixedDepthDrop);

								while (Layers[0]->RefreshingStats.load()) {	std::this_thread::yield(); }
								Layers[0]->Fwd.store(true);
								const auto timePointLocal = timer.now();
								auto SampleLabels = TrainBatch(SampleIndex, N);
								Layers[0]->fpropTime = timer.now() - timePointLocal;
								Layers[0]->Fwd.store(false);

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									if (!Layers[i]->Skip && TaskState.load() == TaskStates::Running)
									{
										while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
										Layers[i]->Fwd.store(true);
										timePoint = timer.now();
										Layers[i]->ForwardProp(N, true);
										Layers[i]->fpropTime = timer.now() - timePoint;
										Layers[i]->Fwd.store(false);
									}
									else
										Layers[i]->fpropTime = std::chrono::duration<Float>(Float(0));
								}
								
								overflow = SampleIndex >= TrainOverflowCount;
								CostFunctionBatch(State.load(), N, overflow, TrainSkipCount);
								RecognizedBatch(State.load(), N, overflow, TrainSkipCount, SampleLabels);
								fpropTime = timer.now() - timePointLocal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (TaskState.load() == TaskStates::Running)
									{
										Layers[i]->bpropTime = std::chrono::duration<Float>(Float(0));
										Layers[i]->updateTime = std::chrono::duration<Float>(Float(0));

										if (!Layers[i]->Skip)
										{
											while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
											Layers[i]->Bwd.store(true);
											timePoint = timer.now();

											if (Layers[i]->HasWeights)
											{
												Layers[i]->ResetGradients();
												Layers[i]->BackwardProp(N);
												Layers[i]->bpropTime = timer.now() - timePoint;

												timePoint = timer.now();
												Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
												Layers[i]->updateTime = timer.now() - timePoint;
									     
												updateTimeCount += Layers[i]->updateTime;
											}
											else
											{
												Layers[i]->BackwardProp(N);
												Layers[i]->bpropTime = timer.now() - timePoint;
											}

											bpropTimeCount += Layers[i]->bpropTime;
											Layers[i]->Bwd.store(false);
										}										
									}
								}
								bpropTime = bpropTimeCount;
								updateTime = updateTimeCount;

								elapsedTime = timer.now() - timePointLocal;
								SampleSpeed = N / (Float(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count()) / 1000000);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
#ifdef DNN_STOCHASTIC
						}
#endif
					}
					else
						break;

					if (CheckTaskState())
					{
						State.store(States::Testing);
#ifdef DNN_STOCHASTIC	
						if (N == 1)
						{
							for (SampleIndex = 0; SampleIndex < DataProv->TestSamplesCount; SampleIndex++)
							{
								auto SampleLabel = TestSample(SampleIndex);

								for (auto cost : CostLayers)
									cost->SetSampleLabel(SampleLabel);

								for (auto i = 1u; i < Layers.size(); i++)
									Layers[i]->ForwardProp(1, false);

								CostFunction(State.load());
								Recognized(State.load(), SampleLabel);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
						}
						else
						{
#endif
							auto overflow = false;
							for (SampleIndex = 0; SampleIndex < AdjustedTestSamplesCount; SampleIndex += N)
							{
								const auto timePointLocal = timer.now();
								while (Layers[0]->RefreshingStats.load()) { std::this_thread::yield(); }
								Layers[0]->Fwd.store(true);
								timePoint = timer.now();
								auto SampleLabels = TestBatch(SampleIndex, N);
								Layers[0]->fpropTime = timer.now() - timePoint;
								Layers[0]->Fwd.store(false);

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
									Layers[i]->Fwd.store(true);
									timePoint = timer.now();
									Layers[i]->ForwardProp(N, false);
									Layers[i]->fpropTime = timer.now() - timePoint;
									Layers[i]->Fwd.store(false);
								}

								fpropTime = timer.now() - timePointLocal;

								overflow = SampleIndex >= TestOverflowCount;
								CostFunctionBatch(State.load(), N, overflow, TestSkipCount);
								RecognizedBatch(State.load(), N, overflow, TestSkipCount, SampleLabels);

								elapsedTime = timer.now() - timePointLocal;
								SampleSpeed = N / (Float(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count()) / 1000000ll);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
#ifdef DNN_STOCHASTIC
						}
#endif
						if (CheckTaskState())
						{
							for (auto cost : CostLayers)
							{
								cost->AvgTrainLoss = cost->TrainLoss / DataProv->TrainSamplesCount;
								cost->AvgTestLoss = cost->TestLoss / DataProv->TestSamplesCount;
								cost->TrainErrorPercentage = cost->TrainErrors / Float(DataProv->TrainSamplesCount / 100);
								cost->TestErrorPercentage = cost->TestErrors / Float(DataProv->TestSamplesCount / 100);
							}

							TrainLoss = CostLayers[CostIndex]->TrainLoss;
							TrainErrors = CostLayers[CostIndex]->TrainErrors;
							AvgTrainLoss = CostLayers[CostIndex]->AvgTrainLoss;
							TrainErrorPercentage = CostLayers[CostIndex]->TrainErrorPercentage;
							TestLoss = CostLayers[CostIndex]->TestLoss;
							TestErrors = CostLayers[CostIndex]->TestErrors;
							AvgTestLoss = CostLayers[CostIndex]->AvgTestLoss;
							TestErrorPercentage = CostLayers[CostIndex]->TestErrorPercentage;
							TrainAccuracy = Float(100) - TrainErrorPercentage;
							TestAccuracy = Float(100) - TestErrorPercentage;

							for (auto& layer : Layers)
								if (layer->CheckOptimizer(Optimizer))
								{
									for (auto& l : Layers)
										l->ResetOptimizer(Optimizer);
									State.store(States::Completed);
									return;
								}

							// save the weights and definition
							State.store(States::SaveWeights);
							const auto epoch = 
								std::string("(") + 
								StringToLower(std::string(magic_enum::enum_name<Datasets>(Dataset))) + 
								std::string(")(") + 
								StringToLower(std::string(magic_enum::enum_name<Optimizers>(Optimizer))) + 
								std::string(")") + 
								std::to_string(CurrentEpoch) + 
								std::string("-") + 
								std::to_string(CurrentCycle) + 
								std::string("-") + 
								std::to_string(TrainErrors) + 
								std::string("-") + 
								std::to_string(TestErrors);

							const auto dir = DataProv->StorageDirectory / std::string("definitions") / Name;
							std::filesystem::create_directories(dir);
							std::filesystem::current_path(dir);
							const auto subdir = dir / epoch;
							std::filesystem::create_directories(subdir);
							SaveWeights((subdir / GetWeightsFileName(PersistOptimizer, Dataset, Optimizer)).string(), PersistOptimizer);
							SaveDefinition((subdir / std::string("model.txt")).string());
							SaveModel((subdir / std::string("model.bin")).string());
							
							State.store(States::NewEpoch);
							const auto dur = timer.now() - timePointGlobal;
							const auto [hrs, mins, secs, ms] = ChronoBurst(dur);
							auto logInfo = LogRecord{};
							logInfo.AutoAugment = CurrentTrainingRate.AutoAugment;
							logInfo.AvgTestLoss = AvgTestLoss;
							logInfo.AvgTrainLoss = AvgTrainLoss;
							logInfo.Beta2 = CurrentTrainingRate.Beta2;
							logInfo.ColorAngle = CurrentTrainingRate.ColorAngle;
							logInfo.ColorCast = CurrentTrainingRate.ColorCast;
							logInfo.CostIndex = CostIndex;
							logInfo.CostName = CostLayers[CostIndex]->Name;
							logInfo.CutMix = CurrentTrainingRate.CutMix;
							logInfo.Cutout = CurrentTrainingRate.Cutout;
							logInfo.Cycle = CurrentCycle;
							logInfo.D = CurrentTrainingRate.D;;
							logInfo.Distortion = CurrentTrainingRate.Distortion;
							logInfo.Dropout = CurrentTrainingRate.Dropout;
							logInfo.ElapsedMilliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
							logInfo.ElapsedTime = (hrs.count() < 10 ? std::string("0") : std::string("")) + std::to_string(hrs.count()) + std::string(":") + (mins.count() < 10 ? std::string("0") : std::string("")) + std::to_string(mins.count()) + std::string(":") + (secs.count() < 10 ? std::string("0") : std::string("")) + std::to_string(secs.count()) + std::string(".") + std::to_string(ms.count());
							logInfo.Epoch = CurrentEpoch;
							logInfo.Eps = CurrentTrainingRate.Eps;
							logInfo.Gamma = CurrentTrainingRate.Gamma;
							logInfo.GroupIndex = GroupIndex;
							logInfo.H = CurrentTrainingRate.H;
							logInfo.HorizontalFlip = CurrentTrainingRate.HorizontalFlip;
							logInfo.InputDropout = CurrentTrainingRate.InputDropout;
							logInfo.Interpolation = CurrentTrainingRate.Interpolation;
							logInfo.L2Penalty = CurrentTrainingRate.L2Penalty;
							logInfo.Momentum = CurrentTrainingRate.Momentum;
							logInfo.N = CurrentTrainingRate.N;
							logInfo.Optimizer = CurrentTrainingRate.Optimizer;
							logInfo.PadD = CurrentTrainingRate.PadD;
							logInfo.PadH = CurrentTrainingRate.PadH;
							logInfo.PadW = CurrentTrainingRate.PadW;
							logInfo.Rate = Rate;
							logInfo.Rotation = CurrentTrainingRate.Rotation;
							logInfo.Scaling = CurrentTrainingRate.Scaling;
							logInfo.TestAccuracy = Float(100) - TestErrorPercentage;
							logInfo.TestErrorPercentage = TestErrorPercentage;
							logInfo.TestErrors = TestErrors;
							logInfo.TrainAccuracy = Float(100) - TrainErrorPercentage;
							logInfo.TrainErrorPercentage = TrainErrorPercentage;
							logInfo.TrainErrors = TrainErrors;
							logInfo.VerticalFlip = CurrentTrainingRate.VerticalFlip;
							logInfo.W = CurrentTrainingRate.W;

							TrainingLog.push_back(logInfo);

							SaveLog((subdir / std::string("log.csv")).string());
							std::filesystem::create_directories(DataProv->StorageDirectory / std::string("state"));
							SaveLog((DataProv->StorageDirectory / std::string("state") / GetLogFileName(Name, Dataset)).string());

							NewEpoch(CurrentCycle, CurrentEpoch, TotalEpochs, static_cast<UInt>(CurrentTrainingRate.Optimizer), CurrentTrainingRate.Beta2, CurrentTrainingRate.Gamma, CurrentTrainingRate.Eps, CurrentTrainingRate.HorizontalFlip, CurrentTrainingRate.VerticalFlip, CurrentTrainingRate.InputDropout, CurrentTrainingRate.Cutout, CurrentTrainingRate.CutMix, CurrentTrainingRate.AutoAugment, CurrentTrainingRate.ColorCast, CurrentTrainingRate.ColorAngle, CurrentTrainingRate.Distortion, static_cast<UInt>(CurrentTrainingRate.Interpolation), CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, CurrentTrainingRate.MaximumRate, CurrentTrainingRate.N, CurrentTrainingRate.D, CurrentTrainingRate.H, CurrentTrainingRate.W, CurrentTrainingRate.PadD, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW, CurrentTrainingRate.Momentum, CurrentTrainingRate.L2Penalty, CurrentTrainingRate.Dropout, AvgTrainLoss, TrainErrorPercentage, Float(100) - TrainErrorPercentage, TrainErrors, AvgTestLoss, TestErrorPercentage, Float(100) - TestErrorPercentage, TestErrors, UInt(dur.count()));
						}
						else
							break;
					}
					else
						break;
				}

				State.store(States::Completed);
			}
		}

#ifndef NDEBUG
		void CheckValid()
		{
			if (TaskState == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);
				//const auto totalSkipConnections = GetTotalSkipConnections();

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				auto bpropTimeCount = std::chrono::duration<Float>(Float(0));
				auto updateTimeCount = std::chrono::duration<Float>(Float(0));
				auto elapsedTime = std::chrono::duration<Float>(Float(0));

				TotalEpochs = 0;
				for (const auto& rate : TrainingRates)
					TotalEpochs += rate.Epochs;
				TotalEpochs += GotoEpoch - 1;

				auto useCycli = false;
				for (const auto& rate : TrainingRates)
					if (rate.Cycles != 1)
						useCycli = true;

				CurrentEpoch = GotoEpoch - 1;
				CurrentTrainingRate = TrainingRates[0];
				Rate = CurrentTrainingRate.MaximumRate;
				CurrentCycle = CurrentTrainingRate.Cycles;

				if (!ChangeResolution(CurrentTrainingRate.N, CurrentTrainingRate.D, CurrentTrainingRate.H, CurrentTrainingRate.W, CurrentTrainingRate.PadD, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
					return;

				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, N);

				//auto learningRateEpochs = CurrentTrainingRate.Epochs;
				//auto learningRateIndex = 0ull;

				RandomTrainSamples = std::vector<UInt>(DataProv->TrainSamplesCount);
				for (auto i = 0ull; i < DataProv->TrainSamplesCount; i++)
					RandomTrainSamples[i] = i;

				TrainSamplesFlip = std::vector<Flip>();
				TestSamplesFlip = std::vector<Flip>();
				for (auto index = 0ull; index < DataProv->TrainSamplesCount; index++)
					TrainSamplesFlip.push_back(Flip{ });
				for (auto index = 0ull; index < DataProv->TestSamplesCount; index++)
					TestSamplesFlip.push_back(Flip{ });

				SetOptimizer(CurrentTrainingRate.Optimizer);
				if (!PersistOptimizer)
					for (auto& layer : Layers)
						layer->ResetOptimizer(Optimizer);
				else
					for (auto& layer : Layers)
						layer->CheckOptimizer(Optimizer);

				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}

				State.store(States::Training);

				if (CheckTaskState())
				{
					for (auto cost : CostLayers)
						cost->Reset();

					auto overflow = false;
					SampleIndex = 0;

					timePointGlobal = timer.now();

					auto SampleLabels = TrainCheckBatch(SampleIndex, N);
					std::filesystem::path path = std::filesystem::path("C:\\Users\\dhaen\\");

					auto os = std::ofstream((path / "testB.bin").string(), std::ios::out | std::ios::binary | std::ios::trunc);
					auto oss = std::ofstream((path / "testB.txt").string(), std::ios::out | std::ios::trunc);

					//size_t point = 0;
					if (!os.bad() && os.is_open())
					{
						Layers[0]->SaveNeurons(os);
						//oss <<  std::to_string(point) + " Input\n";
						//point += Layers[0]->Neurons.size();

						Layers[0]->fpropTime = timer.now() - timePointGlobal;

						//point += Layers[0]->Neurons.size();

						for (auto cost : CostLayers)
							cost->SetSampleLabels(SampleLabels);

						for (auto i = 1ull; i < Layers.size(); i++)
						{
							timePoint = timer.now();
							Layers[i]->ForwardProp(N, true);
							if (i < 3ull)
								Layers[i]->SaveNeurons(os);
							//oss << std::to_string(point) + " " + Layers[i]->Name + "\n";
							//point += Layers[i]->Neurons.size();
							Layers[i]->fpropTime = timer.now() - timePoint;
						}
						fpropTime = timer.now() - timePointGlobal;
						overflow = SampleIndex >= TrainOverflowCount;
						CostFunctionBatch(State.load(), N, overflow, TrainSkipCount);
						RecognizedBatch(State.load(), N, overflow, TrainSkipCount, SampleLabels);

						for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
						{
							if (Layers[i]->HasWeights)
							{
								//Layers[i]->ResetGradients();
								Layers[i]->BackwardProp(N);
								//Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
								//Layers[i]->SaveGradients(os);
							}
							else
								Layers[i]->BackwardProp(N);

							//Layers[i]->SaveNeuronsD1(os);
							//oss << std::to_string(point) + " " + Layers[i]->Name + "\n";
							//point += Layers[i]->NeuronsD1.size();
						}

						os.flush();
						os.close();

						oss.flush();
						oss.close();
					}

					SampleSpeed = N / (Float(std::chrono::duration_cast<std::chrono::microseconds>(fpropTime).count()) / 1000000);

					for (auto cost : CostLayers)
					{
						cost->AvgTestLoss = cost->TrainLoss / DataProv->TrainSamplesCount;
						cost->TestErrorPercentage = cost->TrainErrors / Float(DataProv->TrainSamplesCount / 100);
					}
				}

				TestLoss = CostLayers[CostIndex]->TestLoss;
				AvgTestLoss = CostLayers[CostIndex]->AvgTestLoss;
				TestErrors = CostLayers[CostIndex]->TestErrors;
				TestErrorPercentage = CostLayers[CostIndex]->TestErrorPercentage;
				TestAccuracy = Float(100) - TestErrorPercentage;

				/*
				auto fileName = std::string("C:\\test.txt");
				auto ofs = std::ofstream(fileName);
				if (!ofs.bad())
				{
					for (auto i = 0ull; i < 10000ull; i++)
						ofs << labels[i] << std::endl;
					ofs.flush();
					ofs.close();
				}
				*/

				State.store(States::Completed);;
			}
		}
#endif

		void Testing()
		{
			if (TaskState == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				//auto elapsedTime = std::chrono::duration<Float>(Float(0));

				CurrentTrainingRate = TrainingRates[0];
				Rate = CurrentTrainingRate.MaximumRate;

				if (!ChangeResolution(CurrentTrainingRate.N, CurrentTrainingRate.D, CurrentTrainingRate.H, CurrentTrainingRate.W, CurrentTrainingRate.PadD, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
				{
					State.store(States::Completed);
					return;
				}

				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, N);


				TrainSamplesFlip = std::vector<Flip>();
				TestSamplesFlip = std::vector<Flip>();
				for (auto index = 0ull; index < DataProv->TrainSamplesCount; index++)
					TrainSamplesFlip.push_back(Flip{ Bernoulli<bool>(Float(0.5)), Bernoulli<bool>(Float(0.5)) });
				for (auto index = 0ull; index < DataProv->TestSamplesCount; index++)
					TestSamplesFlip.push_back(Flip{ Bernoulli<bool>(Float(0.5)), Bernoulli<bool>(Float(0.5)) });

				State.store(States::Testing);

				if (CheckTaskState())
				{
					for (auto cost : CostLayers)
						cost->Reset();

#ifdef DNN_STOCHASTIC
					if (N == 1)
					{
						for (SampleIndex = 0; SampleIndex < DataProv->TestSamplesCount; SampleIndex++)
						{
							auto SampleLabel = TestAugmentedSample(SampleIndex);

							for (auto cost : CostLayers)
								cost->SetSampleLabel(SampleLabel);

							for (auto i = 1ull; i < Layers.size(); i++)
								Layers[i]->ForwardProp(1, false);

							CostFunction(State.load());
							Recognized(State.load(), SampleLabel);

							if (TaskState.load() != TaskStates::Running && !CheckTaskState())
								break;
						}
					}
					else
					{
#endif
						auto overflow = false;
						for (SampleIndex = 0; SampleIndex < AdjustedTestSamplesCount; SampleIndex += N)
						{
							timePointGlobal = timer.now();

							while (Layers[0]->RefreshingStats.load()) { std::this_thread::yield(); }
							Layers[0]->Fwd.store(true);
							auto SampleLabels = TestAugmentedBatch(SampleIndex, N);
							Layers[0]->fpropTime = timer.now() - timePointGlobal;
							Layers[0]->Fwd.store(false);

							for (auto cost : CostLayers)
								cost->SetSampleLabels(SampleLabels);

							for (auto i = 1ull; i < Layers.size(); i++)
							{
								while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
								Layers[i]->Fwd.store(true);
								timePoint = timer.now();
								Layers[i]->ForwardProp(N, false);
								Layers[i]->fpropTime = timer.now() - timePoint;
								Layers[i]->Fwd.store(false);
							}

							overflow = SampleIndex >= TestOverflowCount;
							CostFunctionBatch(State.load(), N, overflow, TestSkipCount);
							RecognizedBatch(State.load(), N, overflow, TestSkipCount, SampleLabels);

							fpropTime = timer.now() - timePointGlobal;

							SampleSpeed = N / (Float(std::chrono::duration_cast<std::chrono::microseconds>(fpropTime).count()) / 1000000);

							if (TaskState.load() != TaskStates::Running && !CheckTaskState())
								break;
						}
#ifdef DNN_STOCHASTIC
					}
#endif
					for (auto cost : CostLayers)
					{
						cost->AvgTestLoss = cost->TestLoss / DataProv->TestSamplesCount;
						cost->TestErrorPercentage = cost->TestErrors / Float(DataProv->TestSamplesCount / 100);
					}
				}
			
				TestLoss = CostLayers[CostIndex]->TestLoss;
				AvgTestLoss = CostLayers[CostIndex]->AvgTestLoss;
				TestErrors = CostLayers[CostIndex]->TestErrors;
				TestErrorPercentage = CostLayers[CostIndex]->TestErrorPercentage;
				TestAccuracy = Float(100) - TestErrorPercentage;

				/*
				auto fileName = std::string("C:\\test.txt");
				auto ofs = std::ofstream(fileName);
				if (!ofs.bad())
				{
					for (auto i = 0ull; i < 10000ull; i++)
						ofs << labels[i] << std::endl;
					ofs.flush();
					ofs.close();
				}
				*/

				State.store(States::Completed);;
			}
		}

		bool GetInputSnapShot(Float* snapshot, UInt* label)
		{
			if (!Layers[0]->Neurons.empty() && !BatchSizeChanging.load() && !ResettingWeights.load() && !Layers[0]->Fwd.load())
			{
				const auto idx = UniformInt<UInt>(0ull, N - 1ull) + SampleIndex;
				const auto size = Layers[0]->CDHW();
				const auto offset = (idx - SampleIndex) * size;

				if (State.load() == States::Training && idx < DataProv->TrainSamplesCount)
				{
					label[LabelIndex] = DataProv->TrainLabels[RandomTrainSamples[idx]][LabelIndex];
					
					for (auto i = 0ull; i < size; i++)
						snapshot[i] = Layers[0]->Neurons[i + offset];

					return true;
				}
				else if (State.load() == States::Testing && idx < DataProv->TestSamplesCount)
				{
					label[LabelIndex] = DataProv->TestLabels[idx][LabelIndex];
					
					for (auto i = 0ull; i < size; i++)
						snapshot[i] = Layers[0]->Neurons[i + offset];

					return true;
				}
			}

			return false;
		}

		std::vector<LabelInfo> GetLabelInfo(std::vector<UInt> labels) const
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<LabelInfo>(hierarchies);

			for (auto hierarchie = 0ull; hierarchie < hierarchies; hierarchie++)
			{
				SampleLabels[hierarchie].LabelA = labels[hierarchie];
				SampleLabels[hierarchie].LabelB = labels[hierarchie];
				SampleLabels[hierarchie].Lambda = Float(1);				
			}

			return SampleLabels;
		}

		std::vector<LabelInfo> GetCutMixLabelInfo(std::vector<UInt> labels, std::vector<UInt> mixLabels, double lambda) const
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<LabelInfo>(hierarchies);

			for (auto hierarchie = 0ull; hierarchie < hierarchies; hierarchie++)
			{
				SampleLabels[hierarchie].LabelA = labels[hierarchie];
				SampleLabels[hierarchie].LabelB = mixLabels[hierarchie];
				SampleLabels[hierarchie].Lambda = Float(lambda);
			}

			return SampleLabels;
		}

#ifdef DNN_STOCHASTIC
		std::vector<LabelInfo> TrainSample(const UInt index)
		{
			const auto rndIndex = RandomTrainSamples[index];
			auto imgByte = DataProv->TrainSamples[rndIndex];

			const auto rndIndexMix = (index + 1 >= DataProv->TrainSamplesCount) ? RandomTrainSamples[1] : RandomTrainSamples[index + 1];
			auto imgByteMix = Image<Byte>(DataProv->TrainSamples[rndIndexMix]);

			auto label = DataProv->TrainLabels[rndIndex];
			auto labelMix = DataProv->TrainLabels[rndIndexMix];
			
			std::vector<LabelInfo> SampleLabel;
		
			auto cutout = false;
			if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
			{
				if (CurrentTrainingRate.CutMix)
				{
					double lambda = BetaDistribution<double>(1, 1);
					imgByte = Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
					SampleLabel = GetCutMixLabelInfo(label, labelMix, lambda);
				}
				else
				{
					SampleLabel = GetLabelInfo(label);
					cutout = true;
				}
			}
			else
				SampleLabel = GetLabelInfo(label);

			if (CurrentTrainingRate.HorizontalFlip && TrainSamplesFlip[rndIndex].Horizontal)
				imgByte = Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TrainSamplesFlip[rndIndex].Vertical)
				imgByte = Image<Byte>::VerticalMirror(imgByte);

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (imgByteMix.D() != D || imgByte.H() != H || imgByte.W() != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (cutout)
				imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (CurrentTrainingRate.InputDropout > Float(0))
				imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestSample(const UInt index)
		{
			auto label = DataProv->TestLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestSamples[index];

			if (imgByte.D() != D || imgByte.H() != H || imgByte.W() != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestAugmentedSample(const UInt index)
		{
			auto label = DataProv->TestLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestSamples[index];

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (CurrentTrainingRate.HorizontalFlip && TestSamplesFlip[index].Horizontal)
				imgByte = Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TestSamplesFlip[index].Vertical)
				imgByte = Image<Byte>::VerticalMirror(imgByte);

			if (imgByte.D() != D || imgByte.H() != H || imgByte.W() != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, static_cast<Interpolations>(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, static_cast<Interpolations>(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
				imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			if (CurrentTrainingRate.InputDropout > Float(0))
				Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}
#endif

		std::vector<std::vector<LabelInfo>> TrainCheckBatch(const UInt index, const UInt batchSize)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TrainSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = std::vector<UInt>(DataProv->TrainLabels[sampleIndex]);
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = Image<Byte>(DataProv->TrainSamples[sampleIndex]);
				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TrainBatch(const UInt index, const UInt batchSize)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i_dynamic(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto randomIndex = (index + batchIndex >= DataProv->TrainSamplesCount) ? RandomTrainSamples[batchIndex] : RandomTrainSamples[index + batchIndex];
				auto imgByte = Image<Byte>(DataProv->TrainSamples[randomIndex]);

				const auto randomIndexMix = (index + batchSize - (batchIndex + 1) >= DataProv->TrainSamplesCount) ? RandomTrainSamples[batchSize - (batchIndex + 1)] : RandomTrainSamples[index + batchSize - (batchIndex + 1)];
				auto imgByteMix = Image<Byte>(DataProv->TrainSamples[randomIndexMix]);

				auto labels = std::vector<UInt>(DataProv->TrainLabels[randomIndex]);
				auto mixLabels = std::vector<UInt>(DataProv->TrainLabels[randomIndexMix]);

				auto cutout = false;
				if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
				{
					if (CurrentTrainingRate.CutMix)
					{
						double lambda = BetaDistribution<double>(1, 1);
						imgByte = Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
						SampleLabels[batchIndex] = GetCutMixLabelInfo(labels, mixLabels, lambda);
					}
					else
					{
						SampleLabels[batchIndex] = GetLabelInfo(labels);
						cutout = true;
					}
				}
				else
					SampleLabels[batchIndex] = GetLabelInfo(labels);

				if (CurrentTrainingRate.HorizontalFlip && TrainSamplesFlip[randomIndex].Horizontal)
					imgByte = Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TrainSamplesFlip[randomIndex].Vertical)
					imgByte = Image<Byte>::VerticalMirror(imgByte);

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (cutout)
					imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

				if (RandomCrop)
					imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.InputDropout > Float(0))
					imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = std::vector<UInt>(DataProv->TestLabels[sampleIndex]);
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = Image<Byte>(DataProv->TestSamples[sampleIndex]);
				
				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestAugmentedBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i_dynamic(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = std::vector<UInt>(DataProv->TestLabels[sampleIndex]);
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = Image<Byte>(DataProv->TestSamples[sampleIndex]);

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (CurrentTrainingRate.HorizontalFlip && TestSamplesFlip[sampleIndex].Horizontal)
					imgByte = Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TestSamplesFlip[sampleIndex].Vertical)
					imgByte = Image<Byte>::VerticalMirror(imgByte);

				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
					imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

				if (RandomCrop)
					imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.InputDropout > Float(0))
					imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}
			
		void ForwardProp(const UInt batchSize)
		{
			for (auto &layer : Layers)
				layer->ForwardProp(batchSize, State.load() == States::Training);
		}

		void BackwardProp(const UInt batchSize)
		{
			for (auto i = Layers.size() - 1; i > 0ull; --i)
			{
				if (Layers[i]->HasWeights && TaskState.load() == TaskStates::Running)
				{
					Layers[i]->ResetGradients();
					Layers[i]->BackwardProp(batchSize);
					if (!DisableLocking)
						Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
				}
				else
					Layers[i]->BackwardProp(batchSize);
			}
		}
		
		bool SaveModel(const std::string& fileName) const
		{
			auto os = std::fstream{ fileName, std::ios::out | std::ios::binary | std::ios::trunc };

			if (!os.bad() && os.is_open())
			{
				try
				{
					bitsery::Serializer<bitsery::OutputBufferedStreamAdapter> serializer{ os };
					serializer.object(*this);
					serializer.adapter().flush();
				}
				catch (std::exception&)
				{
					os.close();
					return false;
				}

				os.close();
				return true;
			}

			return false;
		}

		bool LoadModel(const std::string& fileName)
		{
			auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

			if (!is.bad() && is.is_open())
			{
				try
				{
					auto state = bitsery::quickDeserialization<bitsery::InputStreamAdapter>(is, *this);
					assert(state.first == bitsery::ReaderError::NoError && state.second);
				}
				catch (std::exception&)
				{
					is.close();
					return false;
				}

				is.close();
				return true;
			}

			return false;
		}

		void SaveDefinition(const std::string& fileName)
		{
			auto os = std::fstream{ fileName, std::ios::out | std::ios::trunc };

			if (!os.bad() && os.is_open())
			{
				os << CaseInsensitiveReplace(Definition.begin(), Definition.end(), nwl, std::string("\n"));
				os.close();
			}
		}

		void ClearLog()
		{
			TrainingLog = std::vector<LogRecord>();
		}
		
		bool SaveLog(const std::string& fileName)
		{
			try
			{
				CsvFile csv(fileName);

				// Header
				csv << "Cycle" << "Epoch" << "GroupIndex" << "CostIndex" << "CostName" << "N" << "D" << "H" << "W" << "PadD" << "PadH" << "PadW" << "Optimizer" << "Rate" << "Eps" << "Momentum" << "Beta2" << "Gamma" << "L2Penalty" << "Dropout"  << "InputDropout" << "Cutout" << "CutMix" << "AutoAugment" << "HorizontalFlip" << "VerticalFlip" << "ColorCast" << "ColorAngle" << "Distortion" << "Interpolation" << "Scaling" << "Rotation" << "AvgTrainLoss" << "TrainErrors" << "TrainErrorPercentage" << "TrainAccuracy" << "AvgTestLoss" << "TestErrors" << "TestErrorPercentage" << "TestAccuracy" << "ElapsedMilliSeconds" << "ElapsedTime" << EndRow;
					
				// Data
				for (const LogRecord& r : TrainingLog)
					csv << r.Cycle << r.Epoch << r.GroupIndex << r.CostIndex << r.CostName << r.N << r.D << r.H << r.W << r.PadD << r.PadH << r.PadW << std::string(magic_enum::enum_name<Optimizers>(r.Optimizer)) << r.Rate << r.Eps << r.Momentum << r.Beta2 << r.Gamma << r.L2Penalty << r.Dropout << r.InputDropout << r.Cutout << r.CutMix << r.AutoAugment << r.HorizontalFlip << r.VerticalFlip << r.ColorCast << r.ColorAngle << r.Distortion << std::string(magic_enum::enum_name<Interpolations>(r.Interpolation)) << r.Scaling << r.Rotation << r.AvgTrainLoss << r.TrainErrors << r.TrainErrorPercentage << r.TrainAccuracy << r.AvgTestLoss << r.TestErrors << r.TestErrorPercentage << r.TestAccuracy << r.ElapsedMilliSeconds << r.ElapsedTime << EndRow;

				return true;
			}
			catch (const std::exception&)
			{
			}

			return false;
		}

		bool LoadLog(const std::string& fileName)
		{
			auto headers = std::set<std::string>();
			headers.insert(std::string("Cycle"));
			headers.insert(std::string("Epoch"));
			headers.insert(std::string("GroupIndex"));
			headers.insert(std::string("CostIndex"));
			headers.insert(std::string("CostName"));
			headers.insert(std::string("N"));
			headers.insert(std::string("D"));
			headers.insert(std::string("H"));
			headers.insert(std::string("W"));
			headers.insert(std::string("PadD"));
			headers.insert(std::string("PadH"));
			headers.insert(std::string("PadW"));
			headers.insert(std::string("Optimizer"));
			headers.insert(std::string("Rate"));
			headers.insert(std::string("Eps"));
			headers.insert(std::string("Momentum"));
			headers.insert(std::string("Beta2"));
			headers.insert(std::string("Gamma"));
			headers.insert(std::string("L2Penalty"));
			headers.insert(std::string("Dropout"));
			headers.insert(std::string("InputDropout"));
			headers.insert(std::string("Cutout"));
			headers.insert(std::string("CutMix"));
			headers.insert(std::string("AutoAugment"));
			headers.insert(std::string("HorizontalFlip"));
			headers.insert(std::string("VerticalFlip"));
			headers.insert(std::string("ColorCast"));
			headers.insert(std::string("ColorAngle"));
			headers.insert(std::string("Distortion"));
			headers.insert(std::string("Interpolation"));
			headers.insert(std::string("Scaling"));
			headers.insert(std::string("Rotation"));
			headers.insert(std::string("AvgTrainLoss"));
			headers.insert(std::string("TrainErrors"));
			headers.insert(std::string("TrainErrorPercentage"));
			headers.insert(std::string("TrainAccuracy"));
			headers.insert(std::string("AvgTestLoss"));
			headers.insert(std::string("TestErrors"));
			headers.insert(std::string("TestErrorPercentage"));
			headers.insert(std::string("TestAccuracy"));
			headers.insert(std::string("ElapsedMilliSeconds"));
			headers.insert(std::string("ElapsedTime"));

			const auto delimiter = ';';
			auto tmpLog = std::vector<LogRecord>();
			auto record = std::string("");
			auto counter = 0ull;
			const auto fileContents = ReadFileToString(fileName);
			auto iss = std::istringstream(fileContents);
			while (std::getline(iss, record))
			{
				auto line = std::istringstream(record);
				auto idx = 0;
				auto info = LogRecord{};
				while (std::getline(line, record, delimiter))
				{
					if (counter > 0ull)
					{
						try
						{
							switch (idx)
							{
							case 0:		// Cycle
								info.Cycle = std::stoull(record);
								break;
							case 1:		// Epoch
								info.Epoch = std::stoull(record);
								break;
							case 2:		// GroupIndex
								info.GroupIndex = std::stoull(record);
								break;
							case 3:		// CostIndex
								info.CostIndex = std::stoull(record);
								break;
							case 4:		// CostName
								info.CostName = record;
								break;
							case 5:		// N
								info.N = std::stoull(record);
								break;
							case 6:		// D
								info.D = std::stoull(record);
								break;
							case 7:		// H
								info.H = std::stoull(record);
								break;
							case 8:		// W
								info.W = std::stoull(record);
								break;
							case 9:		// PadD
								info.PadD = std::stoull(record);
								break;
							case 10:	// PadH
								info.PadH = std::stoull(record);
								break;
							case 11:	// PadW
								info.PadW = std::stoull(record);
								break;
							case 12:	// Optimizer
							{
								const auto optimizer = magic_enum::enum_cast<Optimizers>(record);
								if (optimizer.has_value())
									info.Optimizer = optimizer.value();
								else
									info.Optimizer = Optimizers::SGDMomentum;
							}
							break;
							case 13:	// Rate
								info.Rate = StringToFloat(record);
								break;
							case 14:	// Eps
								info.Eps = StringToFloat(record);
								break;
							case 15:	// Momentum
								info.Momentum = StringToFloat(record);
								break;
							case 16:	// Beta2
								info.Beta2 = StringToFloat(record);
								break;
							case 17:	// Gamma
								info.Gamma = StringToFloat(record);
								break;
							case 18:	// L2Penalty
								info.L2Penalty = StringToFloat(record);
								break;
							case 19:	// Dropout
								info.Dropout = StringToFloat(record);
								break;
							case 20:	// InputDropout
								info.InputDropout = StringToFloat(record);
								break;
							case 21:	// Cutout
								info.Cutout = StringToFloat(record);
								break;
							case 22:	// CutMix
								info.CutMix = StringToBool(record);
								break;
							case 23:	// AutoAugment
								info.AutoAugment = StringToFloat(record);
								break;
							case 24:	// HorizontalFlip
								info.HorizontalFlip = StringToBool(record);
								break;
							case 25:	// VerticalFlip
								info.VerticalFlip = StringToBool(record);
								break;
							case 26:	// ColorCast
								info.ColorCast = StringToFloat(record);
								break;
							case 27:	// ColorAngle
								info.ColorAngle = std::stoull(record);
								break;
							case 28:	// Distortion
								info.Distortion = StringToFloat(record);
								break;
							case 29:	// Interpolation
							{
								const auto interpolation = magic_enum::enum_cast<Interpolations>(record);
								if (interpolation.has_value())
									info.Interpolation = interpolation.value();
								else
									info.Interpolation = Interpolations::Linear;
							}
							break;
							case 30:	// Scaling
								info.Scaling = StringToFloat(record);
								break;
							case 31:	// Rotation
								info.Rotation = StringToFloat(record);
								break;
							case 32:	// AvgTrainLoss
								info.AvgTrainLoss = StringToFloat(record);
								break;
							case 33:	// TrainErrors
								info.TrainErrors = std::stoull(record);
								break;
							case 34:	// TrainErrorPercentage
								info.TrainErrorPercentage = StringToFloat(record);
								break;
							case 35:	// TrainAccuracy
								info.TrainAccuracy = StringToFloat(record);
								break;
							case 36:	// AvgTestLoss
								info.AvgTestLoss = StringToFloat(record);
								break;
							case 37:	// TestErrors
								info.TestErrors = std::stoull(record);
								break;
							case 38:	// TestErrorPercentage
								info.TestErrorPercentage = StringToFloat(record);
								break;
							case 39:	// TestAccuracy
								info.TestAccuracy = StringToFloat(record);
								break;
							case 40:	// ElapsedMilliSeconds
								info.ElapsedMilliSeconds = std::stoll(record);
								break;
							case 41:	// ElapsedTime
								info.ElapsedTime = record;
								break;
							default:
								break;
							}
						}
						catch (std::exception&)
						{
							return false;
						}
					}
					else
					{
						// check header is valid
						if (headers.find(record) == headers.end())
						{
							return false;
						}
					}

					idx++;
				}

				if (counter > 0ull)
					tmpLog.push_back(info);

				counter++;
			}
			
			tmpLog.shrink_to_fit();
			TrainingLog = std::vector<LogRecord>(tmpLog);

			return true;
		}

		int SaveWeights(const std::string& fileName, const bool persistOptimizer = false) const
		{
			auto os = std::ofstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!os.bad() && os.is_open())
			{
				for (auto& layer : Layers)
					layer->Save(os, persistOptimizer, Optimizer);

				os.close();

				return 0;
			}

			return -1;
		}

		int LoadWeights(const std::string& fileName, const bool persistOptimizer = false)
		{
			const auto& optimizers = magic_enum::enum_entries<Optimizers>();
			
			auto optimizer = Optimizers::SGD;
			for (const auto& opt : optimizers)
			{
				const auto& optimizerString = std::string("(") + StringToLower(std::string(opt.second)) + std::string(")");
				if (fileName.find(optimizerString) != std::string::npos)
					optimizer = opt.first;
			}
			
			if (GetFileSize(fileName) == GetWeightsSize(persistOptimizer, optimizer))
			{
				SetOptimizer(optimizer);

				auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

				if (!is.bad() && is.is_open())
				{
					for (auto& layer : Layers)
						layer->Load(is, persistOptimizer, Optimizer);

					is.close();

					return 0;
				}
			}

#ifndef NDEBUG
			std::cerr << std::string("Model::LoadWeights(const std::string& fileName, const bool persistOptimizer = false)  -  ") << fileName << std::string(", ") << persistOptimizer << std::string("  -  Could not open the file") << std::endl;
#endif

			return -1;
		}

		int SaveLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer = false) const
		{
			auto os = std::ofstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!os.bad() && os.is_open())
			{
				Layers[layerIndex]->Save(os, persistOptimizer, Optimizer);

				os.close();

				return 0;
			}

			return -1;
		}

		int LoadLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer = false)
		{
			if (GetFileSize(fileName) == Layers[layerIndex]->GetWeightsSize(persistOptimizer, Optimizer))
			{
				auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

				if (!is.bad() && is.is_open())
				{
					Layers[layerIndex]->Load(is, persistOptimizer, Optimizer);

					is.close();

					return 0;
				}
			}

			return -1;
		}
	};
	
	template<typename S>
	void serialize(S& s, Flip& o)
	{
		s.boolValue(o.Horizontal);
		s.boolValue(o.Vertical);
	}

	template<typename S>
	void serialize(S& s, TrainingStrategy& o)
	{
		s.value4b(o.Epochs);
		s.value8b(o.N);
		s.value8b(o.D);
		s.value8b(o.H);
		s.value8b(o.W);
		s.value8b(o.PadD);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.value4b(o.Momentum);
		s.value4b(o.Beta2);
		s.value4b(o.Gamma);
		s.value4b(o.L2Penalty);
		s.value4b(o.Dropout);
		s.boolValue(o.HorizontalFlip);
		s.boolValue(o.VerticalFlip);
		s.value4b(o.InputDropout);
		s.value4b(o.Cutout);
		s.boolValue(o.CutMix);
		s.value4b(o.AutoAugment);
		s.value4b(o.ColorCast);
		s.value8b(o.ColorAngle);
		s.value4b(o.Distortion);
		s.value4b(o.Interpolation);
		s.value4b(o.Scaling);
		s.value4b(o.Rotation);
	}
	
	template<typename S>
	void serialize(S& s, Stats& o)
	{
		s.value4b(o.Mean);
		s.value4b(o.StdDev);
		s.value4b(o.Min);
		s.value4b(o.Max);
	}

	template<typename S>
	void serialize(S& s, TrainingRate& o)
	{
		s.value4b(o.Optimizer);
		s.value4b(o.Momentum);
		s.value4b(o.Beta2);
		s.value4b(o.L2Penalty);
		s.value4b(o.Dropout);
		s.value4b(o.Eps);
		s.value8b(o.N);
		s.value8b(o.D);
		s.value8b(o.H);
		s.value8b(o.W);
		s.value8b(o.PadD);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.value8b(o.Cycles);
		s.value8b(o.Epochs);
		s.value8b(o.EpochMultiplier);
		s.value4b(o.MaximumRate);
		s.value4b(o.MinimumRate);
		s.value4b(o.FinalRate);
		s.value4b(o.Gamma);
		s.value8b(o.DecayAfterEpochs);
		s.value4b(o.DecayFactor);
		s.boolValue(o.HorizontalFlip);
		s.boolValue(o.VerticalFlip);
		s.value4b(o.InputDropout);
		s.value4b(o.Cutout);
		s.boolValue(o.CutMix);
		s.value4b(o.AutoAugment);
		s.value4b(o.ColorCast);
		s.value8b(o.ColorAngle);
		s.value4b(o.Distortion);
		s.value4b(o.Interpolation);
		s.value4b(o.Scaling);
		s.value4b(o.Rotation);
	}

	template<typename S>
	void serialize(S& s, LogRecord& o)
	{
		s.value8b(o.Cycle);
		s.value8b(o.Epoch);
		s.value8b(o.GroupIndex);
		s.value8b(o.CostIndex);
		s.text1b(o.CostName, 256);
		
		s.value8b(o.N);
		s.value8b(o.D);
		s.value8b(o.H);
		s.value8b(o.W);
		s.value8b(o.PadD);
		s.value8b(o.PadH);
		s.value8b(o.PadW);

		s.value4b(o.Optimizer);
		s.value4b(o.Rate);
		s.value4b(o.Eps);
		s.value4b(o.Momentum);
		s.value4b(o.Beta2);
		s.value4b(o.Gamma);
		s.value4b(o.L2Penalty);
		s.value4b(o.Dropout);
			
		s.value4b(o.InputDropout);
		s.value4b(o.Cutout);
		s.boolValue(o.CutMix);
		s.value4b(o.AutoAugment);
		s.boolValue(o.HorizontalFlip);
		s.boolValue(o.VerticalFlip);
		s.value4b(o.ColorCast);
		s.value8b(o.ColorAngle);
		s.value4b(o.Distortion);
		s.value4b(o.Interpolation);
		s.value4b(o.Scaling);
		s.value4b(o.Rotation);

		s.value4b(o.AvgTrainLoss);
		s.value8b(o.TrainErrors);
		s.value4b(o.TrainErrorPercentage);
		s.value4b(o.TrainAccuracy);
		s.value4b(o.AvgTestLoss);
		s.value8b(o.TestErrors);
		s.value4b(o.TestErrorPercentage);
		s.value4b(o.TestAccuracy);

		s.value8b(o.ElapsedMilliSeconds);
		s.text1b(o.ElapsedTime, 128);
	}
	
	template<typename S>
	void serialize(S& s, Model& o)
	{
		s.text1b(o.Name, 128);
		s.text1b(o.Script, 1000000);
		s.text1b(o.Definition, 1000000);
		s.value4b(o.Format);
		s.value4b(o.Dataset);
		//s.value4b<States>(o.State);
		//s.value4b<TaskStates>(o.TaskState);
		s.value4b(o.CostFunc);
		s.value4b(o.Optimizer);
		s.value8b(o.CostIndex);
		s.value8b(o.LabelIndex);
		s.value8b(o.GroupIndex);
		s.value8b(o.TotalCycles);
		s.value8b(o.TotalEpochs);
		s.value8b(o.CurrentCycle);
		s.value8b(o.CurrentEpoch);
		s.value8b(o.SampleIndex);
		s.value8b(o.GotoEpoch);
		s.value8b(o.GotoCycle);
		s.value8b(o.AdjustedTrainSamplesCount);
		s.value8b(o.AdjustedTestSamplesCount);
		s.value8b(o.TrainSkipCount);
		s.value8b(o.TestSkipCount);
		s.value8b(o.TrainOverflowCount);
		s.value8b(o.TestOverflowCount);
		s.value8b(o.N);
		s.value8b(o.C);
		s.value8b(o.D);
		s.value8b(o.H);
		s.value8b(o.W);
		s.value8b(o.PadC);
		s.value8b(o.PadD);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.boolValue(o.MirrorPad);
		s.boolValue(o.RandomCrop);
		s.boolValue(o.MeanStdNormalization);
		s.boolValue(o.FixedDepthDrop);
		s.value4b(o.DepthDrop);
		s.value4b(o.WeightsFiller);
		s.value4b(o.WeightsFillerMode);
		s.value4b(o.WeightsGain);
		s.value4b(o.WeightsScale);
		s.value4b(o.WeightsLRM);
		s.value4b(o.WeightsWDM);
		s.value4b(o.BiasesFiller);
		s.value4b(o.BiasesFillerMode);
		s.value4b(o.BiasesGain);
		s.value4b(o.BiasesScale);
		s.value4b(o.BiasesLRM);
		s.value4b(o.BiasesWDM);
		s.value4b(o.AlphaFiller);
		s.value4b(o.BetaFiller);
		s.value4b(o.BatchNormMomentum);
		s.value4b(o.BatchNormEps);
		s.value4b(o.Dropout);
		s.value8b(o.TrainErrors);
		s.value8b(o.TestErrors);
		s.value4b(o.TrainLoss);
		s.value4b(o.TestLoss);
		s.value4b(o.AvgTrainLoss);
		s.value4b(o.AvgTestLoss);
		s.value4b(o.TrainErrorPercentage);
		s.value4b(o.TestErrorPercentage);
		s.value4b(o.TrainAccuracy);
		s.value4b(o.TestAccuracy);
		s.value4b(o.SampleSpeed);
		s.value4b(o.Rate);
		s.boolValue(o.BatchNormScaling);
		s.boolValue(o.HasBias);
		s.boolValue(o.PersistOptimizer);
		s.boolValue(o.DisableLocking);
		s.object(o.CurrentTrainingRate);
		s.container(o.TrainSamplesFlip, 1000000);
		s.container(o.TestSamplesFlip, 500000);
		s.container8b(o.RandomTrainSamples, 1000000);
		s.container(o.TrainingRates, 1024);
		s.container(o.TrainingStrategies, 1024);
		s.boolValue(o.UseTrainingStrategy);
		s.container(o.TrainingLog, 4096);
		//s.value8b(o.FirstUnlockedLayer);
	}
}