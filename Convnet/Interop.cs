using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Float = System.Single;
using UInt = System.UInt64;


namespace Interop
{
    public enum TaskStates
    {
        Paused = 0,
		Running = 1,
		Stopped = 2
	};

    public enum States
    {
        Idle = 0,
		NewEpoch = 1,
		Testing = 2,
		Training = 3,
		SaveWeights = 4,
		Completed = 5
	};

    public enum Optimizers
    {
        AdaBelief = 0,
        AdaBound = 1,
        AdaBoundW = 2,
        AdaDelta = 3,
        AdaGrad = 4,
        Adam = 5,
        Adamax = 6,
        AdamW = 7,
        AmsBound = 8,
        AmsBoundW = 9,
        NAG = 10,
        RMSProp = 11,
        SGD = 12,
        SGDMomentum = 13,
        SGDW = 14
    };

    public enum Costs
    {
        BinaryCrossEntropy = 0,
		CategoricalCrossEntropy = 1,
		MeanAbsoluteEpsError = 2,
		MeanAbsoluteError = 3,
		MeanSquaredError = 4,
		SmoothHinge = 5
	};

    public enum Fillers
    {
        Constant = 0,
		HeNormal = 1,
		HeUniform = 2,
		LecunNormal = 3,
		LeCunUniform = 4,
		Normal = 5,
		TruncatedNormal = 6,
		Uniform = 7
	};

    public enum FillerModes
    {
        Avg = 0,
		In = 1,
		Out = 2
	};

    public enum LayerTypes
    {
        Activation = 0,
		Add = 1,
		Average = 2,
		AvgPooling = 3,
		BatchNorm = 4,
		BatchNormActivation = 5,
		BatchNormActivationDropout = 6,
		BatchNormRelu = 7,
		ChannelSplit = 8,
		ChannelSplitRatioLeft = 9,
		ChannelSplitRatioRight = 10,
		ChannelZeroPad = 11,
		Concat = 12,
		Convolution = 13,
		ConvolutionTranspose = 14,
		Cost = 15,
		Dense = 16,
		DepthwiseConvolution = 17,
		Divide = 18,
		Dropout = 19,
		GlobalAvgPooling = 20,
		GlobalMaxPooling = 21,
		GroupNorm = 22,
		Input = 23,
		LayerNorm = 24,
		LocalResponseNorm = 25,
		LogSoftmax = 26,
		Max = 27,
		MaxPooling = 28,
		Min = 29,
		Multiply = 30,
		PartialDepthwiseConvolution = 31,
		PRelu = 32,
		Reduction = 33,
		Resampling = 34,
		Shuffle = 35,
		Softmax = 36,
		Substract = 37
	};

    public enum Activations
    {
        Abs = 0,
		ASinh = 1,
		BoundedRelu = 2,
		Clip = 3,
		ClipV2 = 4,			//
		Elu = 5,			//
		Exp = 6,			//
		GeluErf = 7,
		GeluTanh = 8,
		HardSigmoid = 9,
		HardSwish = 10,
		Linear = 11,
		Log = 12,
		LogSigmoid = 13,
		Mish = 14,
		Pow = 15,
		Relu = 16,			//
		Round = 17,
		Selu = 18,
		Sigmoid = 19,		//
		SoftPlus = 20,
		SoftRelu = 21,
		SoftSign = 22,
		Sqrt = 23,			//
		Square = 24,
		Swish = 25,
		Tanh = 26,			//
		TanhExp = 27
	};

    public enum Algorithms
    {
        Linear = 0,
		Nearest = 1
	};

    public enum Datasets
    {
        cifar10 = 0,
		cifar100 = 1,
		fashionmnist = 2,
		mnist = 3,
		tinyimagenet = 4
	};

    public enum Models
    {
        densenet = 0,
		efficientnetv2 = 1,
		mobilenetv3 = 2,
		resnet = 3,
		shufflenetv2 = 4
	};

    public enum Positions
    {
        TopLeft = 0,
		TopRight = 1,
		BottomLeft = 2,
		BottomRight = 3,
		Center = 4
	};

    public enum Interpolations
    {
        Cubic = 0,
		Linear = 1,
		Nearest = 2
	};

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct TrainingRate
    {
        public Optimizers Optimizer;
        public Float Momentum;
        public Float Beta2;
        public Float L2Penalty;
        public Float Dropout;
        public Float Eps;
        public UInt N;
        public UInt D;
        public UInt H;
        public UInt W;
        public UInt PadD;
        public UInt PadH;
        public UInt PadW;
        public UInt Cycles;
        public UInt Epochs;
        public UInt EpochMultiplier;
        public Float MaximumRate;
        public Float MinimumRate;
        public Float FinalRate;
        public Float Gamma;
        public UInt DecayAfterEpochs;
        public Float DecayFactor;
        [MarshalAs(UnmanagedType.U1)]
        public bool HorizontalFlip;
        [MarshalAs(UnmanagedType.U1)]
        public bool VerticalFlip;
        public Float InputDropout;
        public Float Cutout;
        [MarshalAs(UnmanagedType.U1)]
        public bool CutMix;
        public Float AutoAugment;
        public Float ColorCast;
        public UInt ColorAngle;
        public Float Distortion;
        public Interpolations Interpolation;
        public Float Scaling;
        public Float Rotation;

        public TrainingRate() 
        { 
			Optimizer = Optimizers.NAG;
            Momentum = (Float)0.9;
            Beta2 = (Float)0.999;
            L2Penalty = (Float)0.0005;
            Dropout = (Float)0;
            Eps = (Float)1E-08;
            N = 1;
            D = 1;
            H = 32;
            W = 32;
            PadD = 0;
            PadH = 4;
            PadW = 4;
            Cycles = 1;
            Epochs = 200;
            EpochMultiplier = 1;
            MaximumRate = (Float)0.05;
            MinimumRate = (Float)0.0001;
            FinalRate = (Float)0.1;
            Gamma = (Float)0.003;
            DecayAfterEpochs = 1;
            DecayFactor = (Float)1;
            HorizontalFlip = false;
            VerticalFlip = false;
            InputDropout = (Float)0;
            Cutout = (Float)0;
            CutMix = false;
            AutoAugment = (Float)0;
            ColorCast = (Float)0;
            ColorAngle = 0;
            Distortion = (Float)0;
            Interpolation = Interpolations.Cubic;
            Scaling = (Float)10.0;
            Rotation = (Float)10.0;
		}

        public TrainingRate(Optimizers optimizer, Float momentum, Float beta2, Float l2penalty, Float dropout, Float eps, UInt n, UInt d, UInt h, UInt w, UInt padD, UInt padH, UInt padW, UInt cycles, UInt epochs, UInt epochMultiplier, Float maximumRate, Float minimumRate, Float finalRate, Float gamma, UInt decayAfterEpochs, Float decayFactor, bool horizontalFlip, bool verticalFlip, Float inputDropout, Float cutout, bool cutMix, Float autoAugment, Float colorCast, UInt colorAngle, Float distortion, Interpolations interpolation, Float scaling, Float rotation)
        {
            Optimizer = optimizer;
            Momentum = momentum;
            Beta2 = beta2;
            L2Penalty = l2penalty;
            Dropout = dropout;
            Eps = eps;
            N = n;
            D = d;
            H = h;
            W = w;
            PadD = padD;
            PadH = padH;
            PadW = padW;
            Cycles = cycles;
            Epochs = epochs;
            EpochMultiplier = epochMultiplier;
            MaximumRate = maximumRate;
            MinimumRate = minimumRate;
            FinalRate = finalRate;
            Gamma = gamma;
            DecayAfterEpochs = decayAfterEpochs;
            DecayFactor = decayFactor;
            HorizontalFlip = horizontalFlip;
            VerticalFlip =verticalFlip;
            InputDropout =inputDropout;
            Cutout = cutout;
            CutMix = cutMix;
            AutoAugment = autoAugment;
            ColorCast = colorCast;
            ColorAngle = colorAngle;
            Distortion = distortion;
            Interpolation = interpolation;
            Scaling = scaling;
            Rotation = rotation;
        }
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct TrainingStrategy
    {
        public Float Epochs;
        public UInt N;
        public UInt D;
        public UInt H;
        public UInt W;
        public UInt PadD;
        public UInt PadH;
        public UInt PadW;
        public Float Momentum;
        public Float Beta2;
        public Float Gamma;
        public Float L2Penalty;
        public Float Dropout;
        [MarshalAs(UnmanagedType.U1)]
        public bool HorizontalFlip;
        [MarshalAs(UnmanagedType.U1)]
        public bool VerticalFlip;
        public Float InputDropout;
        public Float Cutout;
        [MarshalAs(UnmanagedType.U1)]
        public bool CutMix;
        public Float AutoAugment;
        public Float ColorCast;
        public UInt ColorAngle;
        public Float Distortion;
        public Interpolations Interpolation;
        public Float Scaling;
        public Float Rotation;

        public TrainingStrategy()
        {
            Epochs = 1;
            N = 128;
            D = 1;
            H = 32;
            W = 32;
            PadD = 0;
            PadH = 4;
            PadW = 4;
            Momentum = (Float)0.9;
            Beta2 = (Float)0.999;
            Gamma = (Float)0.003;
            L2Penalty = (Float)0.0005;
            Dropout = (Float)0;
            HorizontalFlip = false;
            VerticalFlip = false;
            InputDropout = (Float)0;
            Cutout = (Float)0;
            CutMix = false;
            AutoAugment = (Float)0;
            ColorCast = (Float)0;
            ColorAngle = 0;
            Distortion = (Float)0;
            Interpolation = Interpolations.Cubic;
            Scaling = (Float)10.0;
            Rotation = (Float)10.0;
		}

		public TrainingStrategy(Float epochs, UInt n, UInt d, UInt h, UInt w, UInt padD, UInt padH, UInt padW, Float momentum, Float beta2, Float gamma, Float l2penalty, Float dropout, bool horizontalFlip, bool verticalFlip, Float inputDropout, Float cutout, bool cutMix, Float autoAugment, Float colorCast, UInt colorAngle, Float distortion, Interpolations interpolation, Float scaling, Float rotation)
        {
            Epochs = epochs;
            N = n;
            D = d;
            H = h;
            W = w;
            PadD = padD;
            PadH = padH;
            PadW = padW;
            Momentum = momentum;
            Beta2 = beta2;
            Gamma = gamma;
            L2Penalty = l2penalty;
            Dropout = dropout;
            HorizontalFlip = horizontalFlip;
            VerticalFlip = verticalFlip;
            InputDropout = inputDropout;
            Cutout = cutout;
            CutMix = cutMix;
            AutoAugment = autoAugment;
            ColorCast = colorCast;
            ColorAngle = colorAngle;
            Distortion = distortion;
            Interpolation = interpolation;
            Scaling = scaling;
            Rotation = rotation;
        }
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct Stats
    {
        public Float Mean;
        public Float StdDev;
        public Float Min;
        public Float Max;

        public Stats()
        {
            Mean = (Float)0;
            StdDev = (Float)0;
            Min = (Float)0;
            Max = (Float)0;
        }

        public Stats(Float mean, Float stddev, Float min, Float max)
        {
            Mean = mean;
			StdDev = stddev;
            Min = min;
            Max = max;
        }
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    struct CheckMsg
    {
        public UInt Row;
        public UInt Column;
        [MarshalAs(UnmanagedType.U1)]
        public bool Error;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string Message;

        public CheckMsg(UInt row = 0, UInt column = 0, string message = "", bool error = true)
        {
            Row = row;
            Column = column;
            Message = message;
            Error = error;
        }
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    struct ModelInfo
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string Name;
        public Datasets Dataset;
        public Costs CostFunction;
        public UInt LayerCount;
        public UInt CostLayerCount;
        public UInt CostIndex;
        public UInt GroupIndex;
        public UInt LabelIndex;
        public UInt Hierarchies;
        public UInt TrainSamplesCount;
        public UInt TestSamplesCount;
        [MarshalAs(UnmanagedType.U1)]
        public bool MeanStdNormalization;
        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 3)]
        public Float[] MeanTrainSet;
        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.R4, SizeConst = 3)]
        public Float[] StdTrainSet;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct TrainingInfo
    {
        public UInt TotalCycles;
        public UInt TotalEpochs;
        public UInt Cycle;
        public UInt Epoch;
        public UInt SampleIndex;
        public Float Rate;
        public Optimizers Optimizer;
        public Float Momentum;
        public Float Beta2;
        public Float Gamma;
        public Float L2Penalty;
        public Float Dropout;
        public UInt BatchSize;
        public UInt Height;
        public UInt Width;
        public UInt PadH;
        public UInt PadW;
        [MarshalAs(UnmanagedType.U1)]
        public bool HorizontalFlip;
        [MarshalAs(UnmanagedType.U1)]
        public bool VerticalFlip;
        public Float InputDropout;
        public Float Cutout;
        [MarshalAs(UnmanagedType.U1)]
        public bool CutMix;
        public Float AutoAugment;
        public Float ColorCast;
        public UInt ColorAngle;
        public Float Distortion;
        public Interpolations Interpolation;
        public Float Scaling;
        public Float Rotation;
        public Float AvgTrainLoss;
        public Float TrainErrorPercentage;
        public UInt TrainErrors;
        public Float AvgTestLoss;
        public Float TestErrorPercentage;
        public UInt TestErrors;
        public Float SampleSpeed;
        public States State;
        public TaskStates TaskState;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct TestingInfo
    {
        public UInt TotalCycles;
        public UInt TotalEpochs;
        public UInt Cycle;
        public UInt Epoch;
        public UInt SampleIndex;
        public UInt BatchSize;
        public UInt Height;
        public UInt Width;
        public UInt PadH;
        public UInt PadW;
        public Float AvgTestLoss;
        public Float TestErrorPercentage;
        public UInt TestErrors;
        public Float SampleSpeed;
        public States State;
        public TaskStates TaskState;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    struct LayerInfo
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string Name;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 2048)]
        public string Description;
        public LayerTypes LayerType;
        public Activations Activation;
        public Algorithms Algorithm;
        public Costs Cost;
        public UInt NeuronCount;
        public UInt WeightCount;
        public UInt BiasesCount;
        public UInt LayerIndex;
        public UInt InputsCount;
        public UInt C;
        public UInt D;
        public UInt H;
        public UInt W;
        public UInt PadD;
        public UInt PadH;
        public UInt PadW;
        public UInt KernelH;
        public UInt KernelW;
        public UInt StrideH;
        public UInt StrideW;
        public UInt DilationH;
        public UInt DilationW;
        public UInt Multiplier;
        public UInt Groups;
        public UInt Group;
        public UInt LocalSize;
        public Float Dropout;
        public Float LabelTrue;
        public Float LabelFalse;
        public Float Weight;
        public UInt GroupIndex;
        public UInt LabelIndex;
        public UInt InputC;
        public Float Alpha;
        public Float Beta;
        public Float K;
        public Float fH;
        public Float fW;
        [MarshalAs(UnmanagedType.U1)]
        public bool HasBias;
        [MarshalAs(UnmanagedType.U1)]
        public bool Scaling;
        [MarshalAs(UnmanagedType.U1)]
        public bool AcrossChannels;
        [MarshalAs(UnmanagedType.U1)]
        public bool Locked;
        [MarshalAs(UnmanagedType.U1)]
        public bool Lockable;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    struct CostInfo
    {
        public UInt TrainErrors;
        public Float TrainLoss;
        public Float AvgTrainLoss;
        public Float TrainErrorPercentage;
        public UInt TestErrors;
        public Float TestLoss;
        public Float AvgTestLoss;
        public Float TestErrorPercentage;
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct StatsInfo
    {
        public Stats NeuronsStats;
        public Stats WeightsStats;
        public Stats BiasesStats;
        public Float FPropLayerTime;
        public Float BPropLayerTime;
        public Float UpdateLayerTime;
        public Float FPropTime;
        public Float BPropTime;
        public Float UpdateTime;
        [MarshalAs(UnmanagedType.U1)]
        public bool Locked;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 2048)]
        public string Description;

        public StatsInfo()
        {
            NeuronsStats = new Stats();
            WeightsStats = new Stats();
            BiasesStats = new Stats();
            Description = "";
        }
    };

    [Serializable()]
    public enum DNNAlgorithms
    {
        Linear = 0,
        Nearest = 1
    };

    [Serializable()]
    public enum DNNInterpolations
    {
        Cubic = 0,
        Linear = 1,
        Nearest = 2
    };

    [Serializable()]
    public enum DNNOptimizers
    {
        AdaBelief = 0,
        AdaBound = 1,
        AdaBoundW = 2,
        AdaDelta = 3,
        AdaGrad = 4,
        Adam = 5,
        Adamax = 6,
        AdamW = 7,
        AmsBound = 8,
        AmsBoundW = 9,
        NAG = 10,
        RMSProp = 11,
        SGD = 12,
        SGDMomentum = 13,
        SGDW = 14
    };

    [Serializable()]
    public enum DNNDatasets
    {
        cifar10 = 0,
        cifar100 = 1,
        fashionmnist = 2,
        mnist = 3,
        tinyimagenet = 4
    };

    [Serializable()]
    public enum DNNScripts
    {
        densenet = 0,
        efficientnetv2 = 1,
        mobilenetv3 = 2,
        resnet = 3,
        shufflenetv2 = 4
    };

    [Serializable()]
    public enum DNNCosts
    {
        BinaryCrossEntropy = 0,
        CategoricalCrossEntropy = 1,
        MeanAbsoluteEpsError = 2,
        MeanAbsoluteError = 3,
        MeanSquaredError = 4,
        SmoothHinge = 5
    };

    [Serializable()]
    public enum DNNFillers
    {
        Constant = 0,
        HeNormal = 1,
        HeUniform = 2,
        LeCunNormal = 3,
        LeCunUniform = 4,
        Normal = 5,
        TruncatedNormal = 6,
        Uniform = 7
    };

    [Serializable()]
    public enum DNNFillerModes
    {
        Avg = 0,
        In = 1,
        Out = 2
    };

    [Serializable()]
    public enum DNNLayerTypes
    {
        Activation = 0,
        Add = 1,
        Average = 2,
        AvgPooling = 3,
        BatchNorm = 4,
        BatchNormActivation = 5,
        BatchNormActivationDropout = 6,
        BatchNormRelu = 7,
        ChannelSplit = 8,
        ChannelSplitRatioLeft = 9,
        ChannelSplitRatioRight = 10,
        ChannelZeroPad = 11,
        Concat = 12,
        Convolution = 13,
        ConvolutionTranspose = 14,
        Cost = 15,
        Dense = 16,
        DepthwiseConvolution = 17,
        Divide = 18,
        Dropout = 19,
        GlobalAvgPooling = 20,
        GlobalMaxPooling = 21,
        GroupNorm = 22,
        Input = 23,
        LayerNorm = 24,
        LocalResponseNorm = 25,
        LogSoftmax = 26,
        Max = 27,
        MaxPooling = 28,
        Min = 29,
        Multiply = 30,
        PartialDepthwiseConvolution = 31,
        PRelu = 32,
        Reduction = 33,
        Resampling = 34,
        Shuffle = 35,
        Softmax = 36,
        Substract = 37
    };

    [Serializable()]
    public enum DNNActivations
    {
        Abs = 0,
        ASinh = 1,
        BoundedRelu = 2,
        Clip = 3,
        ClipV2 = 4,         //
        Elu = 5,            //
        Exp = 6,            //
        GeluErf = 7,
        GeluTanh = 8,
        HardSigmoid = 9,
        HardSwish = 10,
        Linear = 11,
        Log = 12,
        LogSigmoid = 13,
        Mish = 14,
        Pow = 15,
        Relu = 16,          //
        Round = 17,
        Selu = 18,
        Sigmoid = 19,       //
        SoftPlus = 20,
        SoftRelu = 21,
        SoftSign = 22,
        Sqrt = 23,          //
        Square = 24,
        Swish = 25,
        Tanh = 26,          //
        TanhExp = 27
    };

    [Serializable()]
    public enum DNNStates
    {
        Idle = 0,
        NewEpoch = 1,
        Testing = 2,
        Training = 3,
        SaveWeights = 4,
        Completed = 5
    };

    [Serializable()]
    public enum DNNTaskStates
    {
        Paused = 0,
        Running = 1,
        Stopped = 2
    };

    [Serializable()]
    public class DNNStats
    {
        public Float Mean;
        public Float StdDev;
        public Float Min;
        public Float Max;

        public DNNStats(ref Stats stats)
        {
            Mean = stats.Mean;
            StdDev = stats.StdDev;
            Min = stats.Min;
            Max = stats.Max;
        }

        public DNNStats(Float mean, Float stddev, Float min, Float max)
        {
            Mean = mean;
            StdDev = stddev;
            Min = min;
            Max = max;
        }
    };

    [Serializable()]
    public class DNNCostLayer
    {
        public DNNCosts CostFunction;
        public UInt LayerIndex;
        public UInt GroupIndex;
        public UInt LabelIndex;
        public UInt ClassCount;
        public string Name;
        public Float Weight;
        public UInt TrainErrors;
        public Float TrainLoss;
        public Float AvgTrainLoss;
        public Float TrainErrorPercentage;
        public Float TrainAccuracy;
        public UInt TestErrors;
        public Float TestLoss;
        public Float AvgTestLoss;
        public Float TestErrorPercentage;
        public Float TestAccuracy;

        public DNNCostLayer(DNNCosts costFunction, UInt layerIndex, UInt groupIndex, UInt labelIndex, UInt classCount, string name, Float weight)
        {
            CostFunction = costFunction;
            LayerIndex = layerIndex;
            GroupIndex = groupIndex;
            LabelIndex = labelIndex;
            ClassCount = classCount;
            Name = name;
            Weight = weight;

            TrainErrors = 0;
            TrainLoss = (Float)0;
            AvgTrainLoss = (Float)0;
            TrainErrorPercentage = (Float)0;
            TrainAccuracy = (Float)0;

            TestErrors = 0;
            TestLoss = (Float)0;
            AvgTestLoss = (Float)0;
            TestErrorPercentage = (Float)0;
            TestAccuracy = (Float)0;
        }
    };

    [Serializable()]
    public class DNNTrainingRate : INotifyPropertyChanged
    {
        [field: NonSerializedAttribute()]
        public event PropertyChangedEventHandler? PropertyChanged;

        private DNNOptimizers optimizer = DNNOptimizers.NAG;
        private Float momentum = (Float)0.9;
        private Float beta2 = (Float)0.999;
        private Float l2Penalty = (Float)0.0005;
        private Float dropout = (Float)0;
        private Float eps = (Float)1E-08;
        private UInt n = 128;
        private UInt d = 1;
        private UInt h = 32;
        private UInt w = 32;
        private UInt padD = 0;
        private UInt padH = 4;
        private UInt padW = 4;
        private UInt cycles = 1;
        private UInt epochs = 200;
        private UInt epochMultiplier = 1;
        private Float maximumRate = (Float)0.05;
        private Float minimumRate = (Float)0.0001;
        private Float finalRate = (Float)0.1;
        private Float gamma = (Float)0.003;
        private UInt decayAfterEpochs = 1;
        private Float decayFactor = (Float)1;
        private bool horizontalFlip = false;
        private bool verticalFlip = false;
        private Float inputDropout = (Float)0;
        private Float cutout = (Float)0;
        private bool cutMix = false;
        private Float autoAugment = (Float)0;
        private Float colorCast = (Float)0;
        private UInt colorAngle = 0;
        private Float distortion = (Float)0;
        private DNNInterpolations interpolation = DNNInterpolations.Linear;
        private Float scaling = (Float)10;
        private Float rotation = (Float)12;

        public DNNOptimizers Optimizer
        {
            get { return optimizer; }
            set
            {
                if (value == optimizer)
                    return;

                optimizer = value;
                OnPropertyChanged(nameof(Optimizer));
            }
        }
        public Float Momentum
        {
            get { return momentum; }
            set
            {
                if (value == momentum || value < (Float)0 || value > (Float)1)
                    return;

                momentum = value;
                OnPropertyChanged(nameof(Momentum));
            }
        }
        public Float Beta2
        {
            get { return beta2; }
            set
            {
                if (value == beta2 || value < (Float)0 || value > (Float)1)
                    return;

                beta2 = value;
                OnPropertyChanged(nameof(Beta2));
            }
        }
        public Float L2Penalty
        {
            get { return l2Penalty; }
            set
            {
                if (value == l2Penalty || value < (Float)0 || value > (Float)1)
                    return;

                l2Penalty = value;
                OnPropertyChanged(nameof(L2Penalty));
            }
        }
        public Float Dropout
        {
            get { return dropout; }
            set
            {
                if (value == dropout || value < (Float)0 || value > (Float)1)
                    return;

                dropout = value;
                OnPropertyChanged(nameof(Dropout));
            }
        }
        public Float Eps
        {
            get { return eps; }
            set
            {
                if (value == eps || value < (Float)0 || value > (Float)1)
                    return;

                eps = value;
                OnPropertyChanged(nameof(Eps));
            }
        }
        public UInt N
        {
            get { return n; }
            set
            {
                if (value == n && value == 0)
                    return;

                n = value;
                OnPropertyChanged(nameof(N));
            }
        }
        public UInt D
        {
            get { return d; }
            set
            {
                if (value == d && value == 0)
                    return;

                d = value;
                OnPropertyChanged(nameof(D));
            }
        }
        public UInt H
        {
            get { return h; }
            set
            {
                if (value == h && value == 0)
                    return;

                h = value;
                OnPropertyChanged(nameof(H));
            }
        }
        public UInt W
        {
            get { return w; }
            set
            {
                if (value == w && value == 0)
                    return;

                w = value;
                OnPropertyChanged(nameof(W));
            }
        }
        public UInt PadD
        {
            get { return padD; }
            set
            {
                if (value == padD && value == 0)
                    return;

                padD = value;
                OnPropertyChanged(nameof(PadD));
            }
        }
        public UInt PadH
        {
            get { return padH; }
            set
            {
                if (value == padH && value == 0)
                    return;

                padH = value;
                OnPropertyChanged(nameof(PadH));
            }
        }
        public UInt PadW
        {
            get { return padW; }
            set
            {
                if (value == padW && value == 0)
                    return;

                padW = value;
                OnPropertyChanged(nameof(PadW));
            }
        }
        public UInt Cycles
        {
            get { return cycles; }
            set
            {
                if (value == cycles && value == 0)
                    return;

                cycles = value;
                OnPropertyChanged(nameof(Cycles));
            }
        }
        public UInt Epochs
        {
            get { return epochs; }
            set
            {
                if (value == epochs && value == 0)
                    return;

                epochs = value;
                OnPropertyChanged(nameof(Epochs));
            }
        }
        public UInt EpochMultiplier
        {
            get { return epochMultiplier; }
            set
            {
                if (value == epochMultiplier && value == 0)
                    return;

                epochMultiplier = value;
                OnPropertyChanged(nameof(EpochMultiplier));
            }
        }
        public Float MaximumRate
        {
            get { return maximumRate; }
            set
            {
                if (value == maximumRate || value < (Float)0 || value > (Float)1)
                    return;

                maximumRate = value;
                OnPropertyChanged(nameof(MaximumRate));
            }
        }
        public Float MinimumRate
        {
            get { return minimumRate; }
            set
            {
                if (value == minimumRate || value < (Float)0 || value > (Float)1)
                    return;

                minimumRate = value;
                OnPropertyChanged(nameof(MinimumRate));
            }
        }
        public Float FinalRate
        {
            get { return finalRate; }
            set
            {
                if (value == finalRate || value < (Float)0 || value > (Float)1)
                    return;

                finalRate = value;
                OnPropertyChanged(nameof(FinalRate));
            }
        }
        public Float Gamma
        {
            get { return gamma; }
            set
            {
                if (value == gamma || value < (Float)0 || value > (Float)1)
                    return;

                gamma = value;
                OnPropertyChanged(nameof(Gamma));
            }
        }
        public UInt DecayAfterEpochs
        {
            get { return decayAfterEpochs; }
            set
            {
                if (value == decayAfterEpochs && value == 0)
                    return;

                decayAfterEpochs = value;
                OnPropertyChanged(nameof(DecayAfterEpochs));
            }
        }
        public Float DecayFactor
        {
            get { return decayFactor; }
            set
            {
                if (value == decayFactor || value < (Float)0 || value > (Float)1)
                    return;

                decayFactor = value;
                OnPropertyChanged(nameof(DecayFactor));
            }
        }
        public bool HorizontalFlip
        {
            get { return horizontalFlip; }
            set
            {
                if (value == horizontalFlip)
                    return;

                horizontalFlip = value;
                OnPropertyChanged(nameof(HorizontalFlip));
            }
        }
        public bool VerticalFlip
        {
            get { return verticalFlip; }
            set
            {
                if (value == verticalFlip)
                    return;

                verticalFlip = value;
                OnPropertyChanged(nameof(VerticalFlip));
            }
        }
        public Float InputDropout
        {
            get { return inputDropout; }
            set
            {
                if (value == inputDropout || value < (Float)0 || value > (Float)1)
                    return;

                inputDropout = value;
                OnPropertyChanged(nameof(InputDropout));
            }
        }
        public Float Cutout
        {
            get { return cutout; }
            set
            {
                if (value == cutout || value < (Float)0 || value > (Float)1)
                    return;

                cutout = value;
                OnPropertyChanged(nameof(Cutout));
            }
        }
        public bool CutMix
        {
            get { return cutMix; }
            set
            {
                if (value == cutMix)
                    return;

                cutMix = value;
                OnPropertyChanged(nameof(CutMix));
            }
        }
        public Float AutoAugment
        {
            get { return autoAugment; }
            set
            {
                if (value == autoAugment || value < (Float)0 || value > (Float)1)
                    return;

                autoAugment = value;
                OnPropertyChanged(nameof(AutoAugment));
            }
        }
        public Float ColorCast
        {
            get { return colorCast; }
            set
            {
                if (value == colorCast || value < (Float)0 || value > (Float)1)
                    return;

                colorCast = value;
                OnPropertyChanged(nameof(ColorCast));
            }
        }
        public UInt ColorAngle
        {
            get { return colorAngle; }
            set
            {
                if (value == colorAngle || value > (Float)360)
                    return;

                colorAngle = value;
                OnPropertyChanged(nameof(ColorAngle));
            }
        }
        public Float Distortion
        {
            get { return distortion; }
            set
            {
                if (value == distortion || value < (Float)0 || value > (Float)1)
                    return;

                distortion = value;
                OnPropertyChanged(nameof(Distortion));
            }
        }
        public DNNInterpolations Interpolation
        {
            get { return interpolation; }
            set
            {
                if (value == interpolation)
                    return;

                interpolation = value;
                OnPropertyChanged(nameof(Interpolation));
            }
        }
        public Float Scaling
        {
            get { return scaling; }
            set
            {
                if (value == scaling || value <= (Float)0 || value > (Float)200)
                    return;

                scaling = value;
                OnPropertyChanged(nameof(Scaling));
            }
        }
        public Float Rotation
        {
            get { return rotation; }
            set
            {
                if (value == rotation || value < (Float)0 || value > (Float)360)
                    return;

                rotation = value;
                OnPropertyChanged(nameof(Rotation));
            }
        }


        public DNNTrainingRate()
        {
            optimizer = DNNOptimizers.NAG;
            momentum = (Float)0.9;
            beta2 = (Float)0.999;
            l2Penalty = (Float)0.0005;
            dropout = 0;
            eps = (Float)1E-08;
            n = 128;
            d = 1;
            h = 32;
            w = 32;
            padD = 0;
            padH = 4;
            padW = 4;
            cycles = 1;
            epochs = 200;
            epochMultiplier = 1;
            maximumRate = (Float)0.05;
            minimumRate = (Float)0.0001;
            finalRate = (Float)0.1;
            gamma = (Float)0.003;
            decayAfterEpochs = 1;
            decayFactor = 1;
            horizontalFlip = true;
            verticalFlip = false;
            inputDropout = 0;
            cutout = 0;
            cutMix = false;
            autoAugment = 0;
            colorCast = 0;
            colorAngle = 16;
            distortion = 0;
            interpolation = DNNInterpolations.Linear;
            scaling = 10;
            rotation = 12;
        }

        public DNNTrainingRate(DNNOptimizers optimizer, Float momentum, Float beta2, Float l2penalty, Float dropout, Float eps, UInt n, UInt d, UInt h, UInt w, UInt padD, UInt padH, UInt padW, UInt cycles, UInt epochs, UInt epochMultiplier, Float maximumRate, Float minimumRate, Float finalRate, Float gamma, UInt decayAfterEpochs, Float decayFactor, bool horizontalFlip, bool verticalFlip, Float inputDropout, Float cutout, bool cutMix, Float autoAugment, Float colorCast, UInt colorAngle, Float distortion, DNNInterpolations interpolation, Float scaling, Float rotation)
        {
            Optimizer = optimizer;
            Momentum = momentum;
            Beta2 = beta2;
            L2Penalty = l2penalty;
            Dropout = dropout;
            Eps = eps;
            N = n;
            D = d;
            H = h;
            W = w;
            PadD = padD;
            PadH = padH;
            PadW = padW;
            Cycles = cycles;
            Epochs = epochs;
            EpochMultiplier = epochMultiplier;
            MaximumRate = maximumRate;
            MinimumRate = minimumRate;
            FinalRate = finalRate;
            Gamma = gamma;
            DecayAfterEpochs = decayAfterEpochs;
            DecayFactor = decayFactor;
            HorizontalFlip = horizontalFlip;
            VerticalFlip = verticalFlip;
            InputDropout = inputDropout;
            Cutout = cutout;
            CutMix = cutMix;
            AutoAugment = autoAugment;
            ColorCast = colorCast;
            ColorAngle = colorAngle;
            Distortion = distortion;
            Interpolation = interpolation;
            Scaling = scaling;
            Rotation = rotation;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    };

    [Serializable()]
    public class DNNTrainingStrategy : INotifyPropertyChanged
    {
        [field: NonSerializedAttribute()]
        public virtual event PropertyChangedEventHandler? PropertyChanged;

        private Float epochs = (Float)1;
        private UInt n = 128;
        private UInt d = 1;
        private UInt h = 32;
        private UInt w = 32;
        private UInt padD = 0;
        private UInt padH = 4;
        private UInt padW = 4;
        private Float momentum = (Float)0.9;
        private Float beta2 = (Float)0.999;
        private Float gamma = (Float)0.003;
        private Float l2Penalty = (Float)0.0005;
        private Float dropout = (Float)0;
        private bool horizontalFlip = false;
        private bool verticalFlip = false;
        private Float inputDropout = (Float)0;
        private Float cutout = (Float)0;
        private bool cutMix = false;
        private Float autoAugment = (Float)0;
        private Float colorCast = (Float)0;
        private UInt colorAngle = 0;
        private Float distortion = (Float)0;
        private DNNInterpolations interpolation = DNNInterpolations.Linear;
        private Float scaling = (Float)10;
        private Float rotation = (Float)12;

        public Float Epochs
        {
            get { return epochs; }
            set
            {
                if (value == epochs || value <= 0 || value > 1)
                    return;

                epochs = value;
                OnPropertyChanged(nameof(Epochs));
            }
        }
        public UInt N
        {
            get { return n; }
            set
            {
                if (value == n && value == 0)
                    return;

                n = value;
                OnPropertyChanged(nameof(N));
            }
        }
        public UInt D
        {
            get { return d; }
            set
            {
                if (value == d && value == 0)
                    return;

                d = value;
                OnPropertyChanged(nameof(D));
            }
        }
        public UInt H
        {
            get { return h; }
            set
            {
                if (value == h && value == 0)
                    return;

                h = value;
                OnPropertyChanged(nameof(H));
            }
        }
        public UInt W
        {
            get { return w; }
            set
            {
                if (value == w && value == 0)
                    return;

                w = value;
                OnPropertyChanged(nameof(W));
            }
        }
        public UInt PadD
        {
            get { return padD; }
            set
            {
                if (value == padD && value == 0)
                    return;

                padD = value;
                OnPropertyChanged(nameof(PadD));
            }
        }
        public UInt PadH
        {
            get { return padH; }
            set
            {
                if (value == padH && value == 0)
                    return;

                padH = value;
                OnPropertyChanged(nameof(PadH));
            }
        }
        public UInt PadW
        {
            get { return padW; }
            set
            {
                if (value == padW && value == 0)
                    return;

                padW = value;
                OnPropertyChanged(nameof(PadW));
            }
        }
        public Float Momentum
        {
            get { return momentum; }
            set
            {
                if (value == momentum || value < (Float)0 || value > (Float)1)
                    return;

                momentum = value;
                OnPropertyChanged(nameof(Momentum));
            }
        }
        public Float Beta2
        {
            get { return beta2; }
            set
            {
                if (value == beta2 || value < (Float)0 || value > (Float)1)
                    return;

                beta2 = value;
                OnPropertyChanged(nameof(Beta2));
            }
        }
        public Float Gamma
        {
            get { return gamma; }
            set
            {
                if (value == gamma || value < (Float)0 || value > (Float)1)
                    return;

                gamma = value;
                OnPropertyChanged(nameof(Gamma));
            }
        }
        public Float L2Penalty
        {
            get { return l2Penalty; }
            set
            {
                if (value == l2Penalty || value < (Float)0 || value > (Float)1)
                    return;

                l2Penalty = value;
                OnPropertyChanged(nameof(L2Penalty));
            }
        }
        public Float Dropout
        {
            get { return dropout; }
            set
            {
                if (value == dropout || value < (Float)0 || value > (Float)1)
                    return;

                dropout = value;
                OnPropertyChanged(nameof(Dropout));
            }
        }
        public bool HorizontalFlip
        {
            get { return horizontalFlip; }
            set
            {
                if (value == horizontalFlip)
                    return;

                horizontalFlip = value;
                OnPropertyChanged(nameof(HorizontalFlip));
            }
        }
        public bool VerticalFlip
        {
            get { return verticalFlip; }
            set
            {
                if (value == verticalFlip)
                    return;

                verticalFlip = value;
                OnPropertyChanged(nameof(VerticalFlip));
            }
        }
        public Float InputDropout
        {
            get { return inputDropout; }
            set
            {
                if (value == inputDropout || value < (Float)0 || value > (Float)1)
                    return;

                inputDropout = value;
                OnPropertyChanged(nameof(InputDropout));
            }
        }
        public Float Cutout
        {
            get { return cutout; }
            set
            {
                if (value == cutout || value < (Float)0 || value > (Float)1)
                    return;

                cutout = value;
                OnPropertyChanged(nameof(Cutout));
            }
        }
        public bool CutMix
        {
            get { return cutMix; }
            set
            {
                if (value == cutMix)
                    return;

                cutMix = value;
                OnPropertyChanged(nameof(CutMix));
            }
        }
        public Float AutoAugment
        {
            get { return autoAugment; }
            set
            {
                if (value == autoAugment || value < (Float)0 || value > (Float)1)
                    return;

                autoAugment = value;
                OnPropertyChanged(nameof(AutoAugment));
            }
        }
        public Float ColorCast
        {
            get { return colorCast; }
            set
            {
                if (value == colorCast || value < (Float)0 || value > (Float)1)
                    return;

                colorCast = value;
                OnPropertyChanged(nameof(ColorCast));
            }
        }
        public UInt ColorAngle
        {
            get { return colorAngle; }
            set
            {
                if (value == colorAngle || value < (Float)0 || value > (Float)360)
                    return;

                colorAngle = value;
                OnPropertyChanged(nameof(ColorAngle));
            }
        }
        public Float Distortion
        {
            get { return distortion; }
            set
            {
                if (value == distortion || value < (Float)0 || value > (Float)1)
                    return;

                distortion = value;
                OnPropertyChanged(nameof(Distortion));
            }
        }
        public DNNInterpolations Interpolation
        {
            get { return interpolation; }
            set
            {
                if (value == interpolation)
                    return;

                interpolation = value;
                OnPropertyChanged(nameof(Interpolation));

            }
        }
        public Float Scaling
        {
            get { return scaling; }
            set
            {
                if (value == scaling || value <= (Float)0 || value > (Float)200)
                    return;

                scaling = value;
                OnPropertyChanged(nameof(Scaling));
            }
        }
        public Float Rotation

        {
            get { return rotation; }
            set
            {
                if (value == rotation || value < (Float)0 || value > (Float)360)
                    return;

                rotation = value;
                OnPropertyChanged(nameof(Rotation));
            }
        }

        public DNNTrainingStrategy()
        {
            epochs = (Float)1;
            n = 128;
            d = 1;
            h = 32;
            w = 32;
            padD = 0;
            padH = 4;
            padW = 4;
            momentum = (Float)0.9;
            beta2 = (Float)0.999;
            gamma = (Float)0.003;
            l2Penalty = (Float)0.0005;
            dropout = (Float)0;
            horizontalFlip = true;
            verticalFlip = false;
            inputDropout = (Float)0;
            cutout = (Float)0;
            cutMix = false;
            autoAugment = (Float)0;
            colorCast = (Float)0;
            colorAngle = 16;
            distortion = (Float)0;
            interpolation = DNNInterpolations.Linear;
            scaling = (Float)10;
            rotation = (Float)12;
        }

        public DNNTrainingStrategy(Float epochs, UInt n, UInt d, UInt h, UInt w, UInt padD, UInt padH, UInt padW, Float momentum, Float beta2, Float gamma, Float l2penalty, Float dropout, bool horizontalFlip, bool verticalFlip, Float inputDropout, Float cutout, bool cutMix, Float autoAugment, Float colorCast, UInt colorAngle, Float distortion, DNNInterpolations interpolation, Float scaling, Float rotation)
        {
            Epochs = epochs;
            N = n;
            D = d;
            H = h;
            W = w;
            PadD = padD;
            PadH = padH;
            PadW = padW;
            Momentum = momentum;
            Beta2 = beta2;
            Gamma = gamma;
            L2Penalty = l2penalty;
            Dropout = dropout;
            HorizontalFlip = horizontalFlip;
            VerticalFlip = verticalFlip;
            InputDropout = inputDropout;
            Cutout = cutout;
            CutMix = cutMix;
            AutoAugment = autoAugment;
            ColorCast = colorCast;
            ColorAngle = colorAngle;
            Distortion = distortion;
            Interpolation = interpolation;
            Scaling = scaling;
            Rotation = rotation;
        }
        
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChangedEventHandler? handler = PropertyChanged;
            if (handler != null) 
                handler(this, new PropertyChangedEventArgs(propertyName));
        }
    };

    [Serializable()]
    public class DNNTrainingResult : INotifyPropertyChanged
    {
        [field: NonSerializedAttribute()]
        public virtual event PropertyChangedEventHandler? PropertyChanged;

        public UInt Cycle
        {
            get { return cycle; }
            set
            {
                if (value == cycle)
                    return;

                cycle = value;
                OnPropertyChanged(nameof(Cycle));
            }
        }
        public UInt Epoch
        {
            get { return epoch; }
            set
            {
                if (value == epoch)
                    return;

                epoch = value;
                OnPropertyChanged(nameof(Epoch));
            }
        }
        public UInt GroupIndex
        {
            get { return groupIndex; }
            set
            {
                if (value == groupIndex)
                    return;

                groupIndex = value;
                OnPropertyChanged(nameof(GroupIndex));
            }
        }
        public UInt CostIndex
        {
            get { return costIndex; }
            set
            {
                if (value == costIndex)
                    return;

                costIndex = value;
                OnPropertyChanged(nameof(CostIndex));
            }
        }
        public string CostName
        {
            get { return costName; }
            set
            {
                if (value == costName)
                    return;

                costName = value;
                OnPropertyChanged(nameof(CostName));
            }
        }
        public UInt N
        {
            get { return n; }
            set
            {
                if (value == n)
                    return;

                n = value;
                OnPropertyChanged(nameof(N));
            }
        }
        public UInt D
        {
            get { return d; }
            set
            {
                if (value == d && value == 0)
                    return;

                d = value;
                OnPropertyChanged(nameof(D));
            }
        }
        public UInt H
        {
            get { return h; }
            set
            {
                if (value == h && value == 0)
                    return;

                h = value;
                OnPropertyChanged(nameof(H));
            }
        }
        public UInt W
        {
            get { return w; }
            set
            {
                if (value == w && value == 0)
                    return;

                w = value;
                OnPropertyChanged(nameof(W));
            }
        }
        public UInt PadD
        {
            get { return padD; }
            set
            {
                if (value == padD && value == 0)
                    return;

                padD = value;
                OnPropertyChanged(nameof(PadD));
            }
        }
        public UInt PadH
        {
            get { return padH; }
            set
            {
                if (value == padH && value == 0)
                    return;

                padH = value;
                OnPropertyChanged(nameof(PadH));
            }
        }
        public UInt PadW
        {
            get { return padW; }
            set
            {
                if (value == padW && value == 0)
                    return;

                padW = value;
                OnPropertyChanged(nameof(PadW));
            }
        }
        public DNNOptimizers Optimizer
        {
            get { return optimizer; }
            set
            {
                if (value == optimizer)
                    return;

                optimizer = value;
                OnPropertyChanged(nameof(Optimizer));
            }
        }
        public Float Rate
        {
            get { return rate; }
            set
            {
                if (value == rate)
                    return;

                rate = value;
                OnPropertyChanged(nameof(Rate));
            }
        }
        public Float Eps
        {
            get { return eps; }
            set
            {
                if (value == eps || value < (Float)0 || value > (Float)1)
                    return;

                eps = value;
                OnPropertyChanged(nameof(Eps));
            }
        }
        public Float Momentum
        {
            get { return momentum; }
            set
            {
                if (value == momentum || value < (Float)0 || value > (Float)1)
                    return;

                momentum = value;
                OnPropertyChanged(nameof(Momentum));
            }
        }
        public Float Beta2
        {
            get { return beta2; }
            set
            {
                if (value == beta2 || value < (Float)0 || value > (Float)1)
                    return;

                beta2 = value;
                OnPropertyChanged(nameof(Beta2));
            }
        }
        public Float Gamma
        {
            get { return gamma; }
            set
            {
                if (value == gamma || value < (Float)0 || value > (Float)1)
                    return;

                gamma = value;
                OnPropertyChanged(nameof(Gamma));
            }
        }
        public Float L2Penalty
        {
            get { return l2Penalty; }
            set
            {
                if (value == l2Penalty || value < (Float)0 || value > (Float)1)
                    return;

                l2Penalty = value;
                OnPropertyChanged(nameof(L2Penalty));
            }
        }
        public Float Dropout
        {
            get { return dropout; }
            set
            {
                if (value == dropout || value < (Float)0 || value > (Float)1)
                    return;

                dropout = value;
                OnPropertyChanged(nameof(Dropout));
            }
        }
        public Float InputDropout
        {
            get { return inputDropout; }
            set
            {
                if (value == inputDropout || value < (Float)0 || value > (Float)1)
                    return;

                inputDropout = value;
                OnPropertyChanged(nameof(InputDropout));
            }
        }
        public Float Cutout
        {
            get { return cutout; }
            set
            {
                if (value == cutout || value < (Float)0 || value > (Float)1)
                    return;

                cutout = value;
                OnPropertyChanged(nameof(Cutout));
            }
        }
        public bool CutMix
        {
            get { return cutMix; }
            set
            {
                if (value == cutMix)
                    return;

                cutMix = value;
                OnPropertyChanged(nameof(CutMix));
            }
        }
        public Float AutoAugment
        {
            get { return autoAugment; }
            set
            {
                if (value == autoAugment || value < (Float)0 || value > (Float)1)
                    return;

                autoAugment = value;
                OnPropertyChanged(nameof(AutoAugment));
            }
        }
        public bool HorizontalFlip
        {
            get { return horizontalFlip; }
            set
            {
                if (value == horizontalFlip)
                    return;

                horizontalFlip = value;
                OnPropertyChanged(nameof(HorizontalFlip));
            }
        }
        public bool VerticalFlip
        {
            get { return verticalFlip; }
            set
            {
                if (value == verticalFlip)
                    return;

                verticalFlip = value;
                OnPropertyChanged(nameof(VerticalFlip));
            }
        }
        public Float ColorCast
        {
            get { return colorCast; }
            set
            {
                if (value == colorCast || value < (Float)0 || value > (Float)1)
                    return;

                colorCast = value;
                OnPropertyChanged(nameof(ColorCast));
            }
        }
        public UInt ColorAngle
        {
            get { return colorAngle; }
            set
            {
                if (value == colorAngle || value > (Float)360)
                    return;

                colorAngle = value;
                OnPropertyChanged(nameof(ColorAngle));
            }
        }
        public Float Distortion
        {
            get { return distortion; }
            set
            {
                if (value == distortion || value < (Float)0 || value > (Float)1)
                    return;

                distortion = value;
                OnPropertyChanged(nameof(Distortion));
            }
        }
        public DNNInterpolations Interpolation
        {
            get { return interpolation; }
            set
            {
                if (value == interpolation)
                    return;

                interpolation = value;
                OnPropertyChanged(nameof(Interpolation));
            }
        }
        public Float Scaling
        {
            get { return scaling; }
            set
            {
                if (value == scaling || value <= (Float)0 || value > (Float)200)
                    return;

                scaling = value;
                OnPropertyChanged(nameof(Scaling));
            }
        }
        public Float Rotation
        {
            get { return rotation; }
            set
            {
                if (value == rotation || value < (Float)0 || value > (Float)360)
                    return;

                rotation = value;
                OnPropertyChanged(nameof(Rotation));
            }
        }
        public Float AvgTrainLoss
        {
            get { return avgTrainLoss; }
            set
            {
                if (value == avgTrainLoss)
                    return;

                avgTrainLoss = value;
                OnPropertyChanged(nameof(AvgTrainLoss));
            }
        }
        public UInt TrainErrors
        {
            get { return trainErrors; }
            set
            {
                if (value == trainErrors)
                    return;

                trainErrors = value;
                OnPropertyChanged(nameof(TrainErrors));
            }
        }
        public Float TrainErrorPercentage
        {
            get { return trainErrorPercentage; }
            set
            {
                if (value == trainErrorPercentage)
                    return;

                trainErrorPercentage = value;
                OnPropertyChanged(nameof(TrainErrorPercentage));
            }
        }
        public Float TrainAccuracy
        {
            get { return trainAccuracy; }
            set
            {
                if (value == trainAccuracy)
                    return;

                trainAccuracy = value;
                OnPropertyChanged(nameof(TrainAccuracy));
            }
        }
        public Float AvgTestLoss
        {
            get { return avgTestLoss; }
            set
            {
                if (value == avgTestLoss)
                    return;

                avgTestLoss = value;
                OnPropertyChanged(nameof(AvgTestLoss));
            }
        }
        public UInt TestErrors
        {
            get { return testErrors; }
            set
            {
                if (value == testErrors)
                    return;

                testErrors = value;
                OnPropertyChanged(nameof(TestErrors));
            }
        }
        public Float TestErrorPercentage
        {
            get { return testErrorPercentage; }
            set
            {
                if (value == testErrorPercentage)
                    return;

                testErrorPercentage = value;
                OnPropertyChanged(nameof(TestErrorPercentage));
            }
        }
        public Float TestAccuracy
        {
            get { return testAccuracy; }
            set
            {
                if (value == testAccuracy)
                    return;

                testAccuracy = value;
                OnPropertyChanged(nameof(TestAccuracy));
            }
        }
        public long ElapsedMilliSeconds
        {
            get { return elapsedMilliSeconds; }
            set
            {
                if (value == elapsedMilliSeconds)
                    return;

                elapsedMilliSeconds = value;
                OnPropertyChanged(nameof(ElapsedMilliSeconds));
            }
        }
        public TimeSpan ElapsedTime
        {
            get { return elapsedTime; }
            set
            {
                if (value == elapsedTime)
                    return;

                elapsedTime = value;
                OnPropertyChanged(nameof(ElapsedTime));
            }
        }

        public DNNTrainingResult() 
        { 
        }

        public DNNTrainingResult(UInt cycle, UInt epoch, UInt groupIndex, UInt costIndex, string costName, UInt n, UInt d, UInt h, UInt w, UInt padD, UInt padH, UInt padW, DNNOptimizers optimizer, Float rate, Float eps, Float momentum, Float beta2, Float gamma, Float l2Penalty, Float dropout, Float inputDropout, Float cutout, bool cutMix, Float autoAugment, bool horizontalFlip, bool verticalFlip, Float colorCast, UInt colorAngle, Float distortion, DNNInterpolations interpolation, Float scaling, Float rotation, Float avgTrainLoss, UInt trainErrors, Float trainErrorPercentage, Float trainAccuracy, Float avgTestLoss, UInt testErrors, Float testErrorPercentage, Float testAccuracy, long elapsedMilliSeconds, TimeSpan elapsedTime)
        {
            Cycle = cycle;
            Epoch = epoch;
            GroupIndex = groupIndex;
            CostIndex = costIndex;
            CostName = costName;
            N = n;
            D = d;
            H = h;
            W = w;
            PadD = padD;
            PadH = padH;
            PadW = padW;
            Optimizer = optimizer;
            Rate = rate;
            Eps = eps;
            Momentum = momentum;
            Beta2 = beta2;
            Gamma = gamma;
            L2Penalty = l2Penalty;
            Dropout = dropout;
            InputDropout = inputDropout;
            Cutout = cutout;
            CutMix = cutMix;
            AutoAugment = autoAugment;
            HorizontalFlip = horizontalFlip;
            VerticalFlip = verticalFlip;
            ColorCast = colorCast;
            ColorAngle = colorAngle;
            Distortion = distortion;
            Interpolation = interpolation;
            Scaling = scaling;
            Rotation = rotation;
            AvgTrainLoss = avgTrainLoss;
            TrainErrors = trainErrors;
            TrainErrorPercentage = trainErrorPercentage;
            TrainAccuracy = trainAccuracy;
            AvgTestLoss = avgTestLoss;
            TestErrors = testErrors;
            TestErrorPercentage = testErrorPercentage;
            TestAccuracy = testAccuracy;
            ElapsedMilliSeconds = elapsedMilliSeconds;
            ElapsedTime = elapsedTime;
        }

        private UInt cycle;
        private UInt epoch;
        private UInt groupIndex;
        private UInt costIndex;
        private string costName;
        private UInt n;
        private UInt d;
        private UInt h;
        private UInt w;
        private UInt padD;
        private UInt padH;
        private UInt padW;
        private DNNOptimizers optimizer;
        private Float rate;
        private Float eps;
        private Float momentum;
        private Float beta2;
        private Float gamma;
        private Float l2Penalty;
        private Float dropout;
        private Float inputDropout;
        private Float cutout;
        private bool cutMix;
        private Float autoAugment;
        private bool horizontalFlip;
        private bool verticalFlip;
        private Float colorCast;
        private UInt colorAngle;
        private Float distortion;
        private DNNInterpolations interpolation;
        private Float scaling;
        private Float rotation;
        private Float avgTrainLoss;
        private UInt trainErrors;
        private Float trainErrorPercentage;
        private Float trainAccuracy;
        private Float avgTestLoss;
        private UInt testErrors;
        private Float testErrorPercentage;
        private Float testAccuracy;
        private long elapsedMilliSeconds;
        private TimeSpan elapsedTime;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    };

    [Serializable()]
    public struct DNNCheckMsg(UInt row, UInt column, string message, bool error, string definition)
    {
        public UInt Row = row;
        public UInt Column = column;
        public bool Error = error;
        public string Message = message;
        public string Definition = definition;
    };

    [Serializable()]
    public class DNNLayerInfo : INotifyPropertyChanged
    {
        [field: NonSerializedAttribute()]
        public virtual event PropertyChangedEventHandler? PropertyChanged;

        private string name;
        private string description;
        private DNNLayerTypes layerType;
        private DNNActivations activation;
        private DNNCosts cost;
        private System.Collections.Generic.List<UInt> inputs;
        //private System.Collections.Generic.List<string> InputsNames;
        private Avalonia.Media.Imaging.WriteableBitmap weightsSnapshot;
        private bool lockable;
        private bool? lockUpdate = false;
        private bool isNormLayer;
        private bool hasBias;
        private bool mirrorPad;
        private bool randomCrop;
        private bool scaling;
        private bool acrossChannels;
        private int weightsSnapshotX;
        private int weightsSnapshotY;
        private UInt inputCount;
        private UInt layerIndex;
        private UInt neuronCount;
        private UInt c;
        private UInt d;
        private UInt w;
        private UInt h;
        private UInt kernelH;
        private UInt kernelW;
        private UInt dilationH;
        private UInt dilationW;
        private UInt strideH;
        private UInt strideW;
        private UInt padD;
        private UInt padH;
        private UInt padW;
        private UInt multiplier;
        private UInt groups;
        private UInt group;
        private UInt localSize;
        private UInt weightCount;
        private UInt biasCount;
        private UInt groupSize;
        private UInt inputC;
        private UInt groupIndex;
        private UInt labelIndex;
        private Float dropout;
        private Float cutout;
        private DNNStats neuronsStats;
        private DNNStats weightsStats;
        private DNNStats biasesStats;
        private Float weight;
        private Float alpha;
        private Float beta;
        private Float k;
        private DNNAlgorithms algorithm;
        private Float factorH;
        private Float factorW;
        private Float fPropLayerTime;
        private Float bPropLayerTime;
        private Float updateLayerTime;

        public string Name
        {
            get { return name; }
            set
            {
                if (value == name)
                    return;

                name = value;
                OnPropertyChanged(nameof(Name));
            }
        }
        public string Description
        {
            get { return description; }
            set
            {
                if (value == description)
                    return;

                description = value;
                OnPropertyChanged(nameof(Description));
            }
        }
        public DNNLayerTypes LayerType
        {
            get { return layerType; }
            set
            {
                if (value == layerType)
                    return;

                layerType = value;
                OnPropertyChanged(nameof(LayerType));
            }
        }
        public DNNActivations Activation
        {
            get { return activation; }
            set
            {
                if (value == activation)
                    return;

                activation = value;
                OnPropertyChanged(nameof(Activation));
            }
        }
        public DNNCosts Cost
        {
            get { return cost; }
            set
            {
                if (value == cost)
                    return;

                cost = value;
                OnPropertyChanged(nameof(Cost));
            }
        }
        public System.Collections.Generic.List<UInt> Inputs
        {
            get { return inputs; }
            set
            {
                if (value == inputs)
                    return;

                inputs = value;
                OnPropertyChanged(nameof(Inputs));
            }
        }
        //public System.Collections.Generic.List<string> InputsNames;
        public Avalonia.Media.Imaging.WriteableBitmap WeightsSnapshot
        {
            get { return weightsSnapshot; }
            set
            {
                if (value == weightsSnapshot)
                    return;

                weightsSnapshot = value;
                OnPropertyChanged(nameof(WeightsSnapshot));
            }
        }
        public bool Lockable
        {
            get { return lockable; }
            set
            {
                if (value == lockable)
                    return;

                lockable = value;
                OnPropertyChanged(nameof(Lockable));
            }
        }
        public bool? LockUpdate
        {
            get { return lockUpdate; }
            set
            {
                if (value.Equals(lockUpdate))
                    return;

                lockUpdate = value;
                OnPropertyChanged(nameof(LockUpdate));
            }
        }
        public bool IsNormLayer
        {
            get { return isNormLayer; }
            set
            {
                if (value == isNormLayer)
                    return;

                isNormLayer = value;
                OnPropertyChanged(nameof(IsNormLayer));
            }
        }
        public bool HasBias
        {
            get { return hasBias; }
            set
            {
                if (value == hasBias)
                    return;

                hasBias = value;
                OnPropertyChanged(nameof(HasBias));
            }
        }
        public bool MirrorPad
        {
            get { return mirrorPad; }
            set
            {
                if (value == mirrorPad)
                    return;

                mirrorPad = value;
                OnPropertyChanged(nameof(MirrorPad));
            }
        }
        public bool RandomCrop
        {
            get { return randomCrop; }
            set
            {
                if (value == randomCrop)
                    return;

                randomCrop = value;
                OnPropertyChanged(nameof(RandomCrop));
            }
        }
        public bool Scaling
        {
            get { return scaling; }
            set
            {
                if (value == scaling)
                    return;

                scaling = value;
                OnPropertyChanged(nameof(Scaling));
            }
        }
        public bool AcrossChannels
        {
            get { return acrossChannels; }
            set
            {
                if (value == acrossChannels)
                    return;

                acrossChannels = value;
                OnPropertyChanged(nameof(AcrossChannels));
            }
        }
        public int WeightsSnapshotX
        {
            get { return weightsSnapshotX; }
            set
            {
                if (value == weightsSnapshotX)
                    return;

                weightsSnapshotX = value;
                OnPropertyChanged(nameof(WeightsSnapshotX));
            }
        }
        public int WeightsSnapshotY
        {
            get { return weightsSnapshotY; }
            set
            {
                if (value == weightsSnapshotY)
                    return;

                weightsSnapshotY = value;
                OnPropertyChanged(nameof(WeightsSnapshotY));
            }
        }
        public UInt InputCount
        {
            get { return inputCount; }
            set
            {
                if (value == inputCount)
                    return;

                inputCount = value;
                OnPropertyChanged(nameof(InputCount));
            }
        }
        public UInt LayerIndex
        {
            get { return layerIndex; }
            set
            {
                if (value == layerIndex)
                    return;

                c = value;
                OnPropertyChanged(nameof(LayerIndex));
            }
        }
        public UInt NeuronCount
        {
            get { return neuronCount; }
            set
            {
                if (value == neuronCount)
                    return;

                neuronCount = value;
                OnPropertyChanged(nameof(NeuronCount));
            }
        }
        public UInt C
        {
            get { return c; }
            set
            {
                if (value == c)
                    return;

                c = value;
                OnPropertyChanged(nameof(C));
            }
        }
        public UInt D
        {
            get { return d; }
            set
            {
                if (value == d && value == 0)
                    return;

                d = value;
                OnPropertyChanged(nameof(D));
            }
        }
        public UInt H
        {
            get { return h; }
            set
            {
                if (value == h && value == 0)
                    return;

                h = value;
                OnPropertyChanged(nameof(H));
            }
        }
        public UInt W
        {
            get { return w; }
            set
            {
                if (value == w && value == 0)
                    return;

                w = value;
                OnPropertyChanged(nameof(W));
            }
        }
        public UInt KernelW
        {
            get { return kernelW; }
            set
            {
                if (value == kernelW)
                    return;

                kernelW = value;
                OnPropertyChanged(nameof(KernelW));
            }
        }
        public UInt KernelH
        {
            get { return kernelH; }
            set
            {
                if (value == kernelH)
                    return;

                kernelH = value;
                OnPropertyChanged(nameof(KernelH));
            }
        }
        public UInt DilationW
        {
            get { return dilationW; }
            set
            {
                if (value == dilationW)
                    return;

                dilationW = value;
                OnPropertyChanged(nameof(DilationW));
            }
        }
        public UInt DilationH
        {
            get { return dilationH; }
            set
            {
                if (value == dilationH)
                    return;

                dilationH = value;
                OnPropertyChanged(nameof(DilationH));
            }
        }
        public UInt StrideW
        {
            get { return strideW; }
            set
            {
                if (value == strideW)
                    return;

                strideW = value;
                OnPropertyChanged(nameof(StrideW));
            }
        }
        public UInt StrideH
        {
            get { return strideH; }
            set
            {
                if (value == strideH)
                    return;

                strideH = value;
                OnPropertyChanged(nameof(StrideH));
            }
        }
        public UInt PadD
        {
            get { return padD; }
            set
            {
                if (value == padD && value == 0)
                    return;

                padD = value;
                OnPropertyChanged(nameof(PadD));
            }
        }
        public UInt PadH
        {
            get { return padH; }
            set
            {
                if (value == padH && value == 0)
                    return;

                padH = value;
                OnPropertyChanged(nameof(PadH));
            }
        }
        public UInt PadW
        {
            get { return padW; }
            set
            {
                if (value == padW && value == 0)
                    return;

                padW = value;
                OnPropertyChanged(nameof(PadW));
            }
        }
        public UInt Multiplier
        {
            get { return multiplier; }
            set
            {
                if (value == multiplier)
                    return;

                multiplier = value;
                OnPropertyChanged(nameof(Multiplier));
            }
        }
        public UInt Groups
        {
            get { return groups; }
            set
            {
                if (value == groups)
                    return;

                groups = value;
                OnPropertyChanged(nameof(Groups));
            }
        }
        public UInt Group
        {
            get { return group; }
            set
            {
                if (value == group)
                    return;

                group = value;
                OnPropertyChanged(nameof(C));
            }
        }
        public UInt LocalSize
        {
            get { return localSize; }
            set
            {
                if (value == localSize)
                    return;

                localSize = value;
                OnPropertyChanged(nameof(LocalSize));
            }
        }
        public UInt WeightCount
        {
            get { return weightCount; }
            set
            {
                if (value == weightCount)
                    return;

                weightCount = value;
                OnPropertyChanged(nameof(WeightCount));
            }
        }
        public UInt BiasCount
        {
            get { return biasCount; }
            set
            {
                if (value == biasCount)
                    return;

                biasCount = value;
                OnPropertyChanged(nameof(BiasCount));
            }
        }
        public UInt GroupSize
        {
            get { return groupSize; }
            set
            {
                if (value == groupSize)
                    return;

                groupSize = value;
                OnPropertyChanged(nameof(GroupSize));
            }
        }
        public UInt InputC
        {
            get { return inputC; }
            set
            {
                if (value == inputC)
                    return;

                inputC = value;
                OnPropertyChanged(nameof(InputC));
            }
        }
        public UInt GroupIndex
        {
            get { return groupIndex; }
            set
            {
                if (value == groupIndex)
                    return;

                groupIndex = value;
                OnPropertyChanged(nameof(GroupIndex));
            }
        }
        public UInt LabelIndex
        {
            get { return labelIndex; }
            set
            {
                if (value == labelIndex)
                    return;

                labelIndex = value;
                OnPropertyChanged(nameof(LabelIndex));
            }
        }
        public Float Dropout
        {
            get { return dropout; }
            set
            {
                if (value == dropout)
                    return;

                dropout = value;
                OnPropertyChanged(nameof(Dropout));
            }
        }
        public Float Cutout
        {
            get { return cutout; }
            set
            {
                if (value == cutout)
                    return;

                cutout = value;
                OnPropertyChanged(nameof(Cutout));
            }
        }
        public DNNStats NeuronsStats
        {
            get { return neuronsStats; }
            set
            {
                if (value == neuronsStats)
                    return;

                neuronsStats = value;
                OnPropertyChanged(nameof(NeuronsStats));
            }
        }
        public DNNStats WeightsStats
        {
            get { return weightsStats; }
            set
            {
                if (value == weightsStats)
                    return;

                weightsStats = value;
                OnPropertyChanged(nameof(WeightsStats));
            }
        }
        public DNNStats BiasesStats
        {
            get { return biasesStats; }
            set
            {
                if (value == biasesStats)
                    return;

                biasesStats = value;
                OnPropertyChanged(nameof(BiasesStats));
            }
        }
        public Float Weight
        {
            get { return weight; }
            set
            {
                if (value == weight)
                    return;

                weight = value;
                OnPropertyChanged(nameof(Weight));
            }
        }
        public Float Alpha
        {
            get { return alpha; }
            set
            {
                if (value == alpha)
                    return;

                alpha = value;
                OnPropertyChanged(nameof(Alpha));
            }
        }
        public Float Beta
        {
            get { return beta; }
            set
            {
                if (value == beta)
                    return;

                beta = value;
                OnPropertyChanged(nameof(Beta));
            }
        }
        public Float K
        {
            get { return k; }
            set
            {
                if (value == k)
                    return;

                k = value;
                OnPropertyChanged(nameof(K));
            }
        }
        public DNNAlgorithms Algorithm
        {
            get { return algorithm; }
            set
            {
                if (value == algorithm)
                    return;

                algorithm = value;
                OnPropertyChanged(nameof(Algorithm));
            }
        }
        public Float FactorH
        {
            get { return factorH; }
            set
            {
                if (value == factorH)
                    return;

                factorH = value;
                OnPropertyChanged(nameof(FactorH));
            }
        }
        public Float FactorW
        {
            get { return factorW; }
            set
            {
                if (value == factorW)
                    return;

                factorW = value;
                OnPropertyChanged(nameof(FactorW));
            }
        }
        public Float FPropLayerTime
        {
            get { return fPropLayerTime; }
            set
            {
                if (value == fPropLayerTime)
                    return;

                fPropLayerTime = value;
                OnPropertyChanged(nameof(FPropLayerTime));
            }
        }
        public Float BPropLayerTime
        {
            get { return bPropLayerTime; }
            set
            {
                if (value == bPropLayerTime)
                    return;

                bPropLayerTime = value;
                OnPropertyChanged(nameof(BPropLayerTime));
            }
        }
        public Float UpdateLayerTime
        {
            get { return updateLayerTime; }
            set
            {
                if (value == updateLayerTime)
                    return;

                updateLayerTime = value;
                OnPropertyChanged(nameof(UpdateLayerTime));
            }
        }

        public DNNLayerInfo()
        {
            NeuronsStats = new DNNStats((Float)0, (Float)0, (Float)0, (Float)0);
            WeightsStats = new DNNStats((Float)0, (Float)0, (Float)0, (Float)0);
            BiasesStats = new DNNStats((Float)0, (Float)0, (Float)0, (Float)0);
            Name = "";
            Description = "";
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    public class DNNModel : IDisposable
    {
        private const string library = "dnn";
        private const CharSet charSet = CharSet.Ansi;
        private const UnmanagedType stringType = UnmanagedType.LPStr;
        private const CallingConvention CC = CallingConvention.Cdecl;

        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNDataprovider([MarshalAs(stringType)] string directory);

        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNRead([MarshalAs(stringType)] string definition, [In, Out] ref CheckMsg checkMsg);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNLoadDataset();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetTrainingInfo([In, Out] ref TrainingInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetTestingInfo([In, Out] ref TestingInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetModelInfo([In, Out] ref ModelInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetLayerInfo(UInt layerIndex, [In, Out] ref LayerInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetLayerInputs(UInt layerIndex, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U8, SizeConst = 10, SizeParamIndex = 0)][In,Out] UInt[] inputs);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern Optimizers GetOptimizer();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNModelDispose();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNDataproviderDispose();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNPersistOptimizer(bool persistOptimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNDisableLocking(bool disable);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNSetShuffleCount(UInt count);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNBatchNormUsed();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNStochasticEnabled();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNRefreshStatistics(UInt layerIndex, [In,Out] ref StatsInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNGetInputSnapShot([MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4, SizeParamIndex = 0, SizeConst = 10000000)][In,Out] Float[] snapshot, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U8, SizeParamIndex = 0, SizeConst = 10)][In,Out] UInt[] label);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetImage(UInt layerIndex, Byte fillColor, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U1, SizeConst = 500000000, SizeParamIndex = 0)][In,Out] Byte[] image);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNSetFormat(bool plain);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNSetOptimizer(Optimizers optimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNResetOptimizer();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNSetUseTrainingStrategy(bool enable);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNSetCostIndex(UInt costLayerIndex);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetCostInfo(UInt index, [In, Out] ref CostInfo info);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNAddTrainingRate(ref TrainingRate rate, bool clear, UInt gotoEpoch, UInt trainSamples);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNAddTrainingRateSGDR(ref TrainingRate rate, bool clear, UInt gotoEpoch, UInt gotoCycle, UInt trainSamples);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNClearTrainingStrategies();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNAddTrainingStrategy(ref TrainingStrategy strategy);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        unsafe private static extern void DNNSetNewEpochDelegate(void* newEpoch);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNTraining();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNTesting();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNStop();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNPause();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNResume();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNSetLocked(bool locked);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNSetLayerLocked(UInt layerIndex, bool locked);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNCheck([MarshalAs(stringType)][In, Out] StringBuilder definition, [In, Out] ref CheckMsg checkMsg);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNLoad([MarshalAs(stringType)] string fileName, [In, Out] ref CheckMsg checkMsg);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNResetWeights();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNLoadModel([MarshalAs(stringType)] string fileName);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNSaveModel([MarshalAs(stringType)] string fileName);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNClearLog();
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNLoadLog([MarshalAs(stringType)] string fileName);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern bool DNNSaveLog([MarshalAs(stringType)] string fileName);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNLoadWeights([MarshalAs(stringType)] string fileName, bool persistOptimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNSaveWeights([MarshalAs(stringType)] string fileName, bool persistOptimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNLoadLayerWeights([MarshalAs(stringType)] string fileName, UInt layerIndex, bool persistOptimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern int DNNSaveLayerWeights([MarshalAs(stringType)] string fileName, UInt layerIndex, bool persistOptimizer);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNResetLayerWeights(UInt layerIndex);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetConfusionMatrix(UInt costLayerIndex, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U8, SizeParamIndex = 0, SizeConst = 10000000)][In, Out] UInt[] confusionMatrix);
        [DllImport(library, BestFitMapping = true, CallingConvention = CC, CharSet = charSet, ExactSpelling = true)]
        private static extern void DNNGetLayerWeights(UInt layerIndex, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4, SizeConst = 500000000, SizeParamIndex = 0)] [In,Out] Float[] weights, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4, SizeConst = 5000000, SizeParamIndex = 0)][In, Out] Float[] biases);

        private static Byte FloatSaturate(Float value) => value > (Float)255 ? (Byte)255 : value < (Float)0 ? (Byte)0 : (Byte)value;
        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        public delegate void TrainProgressEventDelegate(DNNOptimizers Optim, UInt BatchSize, UInt Cycle, UInt TotalCycles, UInt Epoch, UInt TotalEpochs, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, DNNInterpolations Interpolation, Float Scaling, Float Rotation, UInt SampleIndex, Float Rate, Float Momentum, Float Beta2, Float Gamma, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, DNNStates State, DNNTaskStates TaskState);
        public delegate void TestProgressEventDelegate(UInt BatchSize, UInt SampleIndex, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, DNNStates State, DNNTaskStates TaskState);
        public delegate void NewEpochEventDelegate(UInt Cycle, UInt Epoch, UInt TotalEpochs, UInt Optimizer, Float Beta2, Float Gamma, Float Eps, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, UInt Interpolation, Float Scaling, Float Rotation, Float Rate, UInt N, UInt D, UInt H, UInt W, UInt PadD, UInt PadH, UInt PadW, Float Momentum, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, UInt ElapsedNanoSecondes);
        
		private readonly System.Timers.Timer WorkerTimer;
		private readonly StringBuilder sb;
		// private string oldWeightSaveFileName;

        public TrainProgressEventDelegate TrainProgress;
	    public TestProgressEventDelegate TestProgress;
		public NewEpochEventDelegate NewEpoch;

		public Byte BackgroundColor;
		public int SelectedIndex;
        public System.Collections.ObjectModel.ObservableCollection<DNNLayerInfo> Layers;
        public Avalonia.Media.Imaging.WriteableBitmap InputSnapshot;
        public string Label;
		public DNNCostLayer[] CostLayers;
        public Float[] MeanTrainSet;
        public Float[] StdTrainSet;
        public UInt[] ConfusionMatrix;
        public string[][] LabelsCollection;
        public bool UseTrainingStrategy;
        public System.Collections.ObjectModel.ObservableCollection<DNNTrainingStrategy>? TrainingStrategies;
        public DNNTrainingRate[] TrainingRates;
        public DNNTrainingRate TrainingRate;
		public string Definition;
		public string StorageDirectory;
		public string DatasetsDirectory;
		public string DefinitionsDirectory;
		public string Name;
		public string DurationString;
		public DNNDatasets Dataset;
		public DNNOptimizers Optimizer;
		public DNNCosts CostFunction;
		public Stopwatch Duration;
		public bool HorizontalFlip;
        public bool VerticalFlip;
        public bool MeanStdNormalization;
        public bool IsTraining;
        public UInt CostIndex;
		public UInt Hierarchies;
		public UInt ClassCount;
		public UInt GroupIndex;
		public UInt LabelIndex;
		public UInt BatchSize;
		public UInt Height;
		public UInt Width;
		public UInt PadH;
		public UInt PadW;
		public UInt LayerCount;
		public UInt Multiplier;
		public UInt CostLayerCount;
		public UInt TrainingSamples;
		public UInt AdjustedTrainSamplesCount;
		public UInt TestingSamples;
		public UInt AdjustedTestSamplesCount;
		public UInt Cycle;
		public UInt TotalCycles;
		public UInt Epoch;
		public UInt TotalEpochs;
		public Float Gamma;
		public Float ColorCast;
		public UInt ColorAngle;
		public DNNInterpolations Interpolation;
		public UInt SampleIndex;
		public UInt TrainErrors;
		public UInt TestErrors;
		public UInt BlockSize;
		public Float InputDropout;
		public Float Cutout;
		public bool CutMix;
        public Float AutoAugment;
		public Float Distortion;
		public Float Scaling;
		public Float Rotation;
		public Float AvgTrainLoss;
		public Float TrainErrorPercentage;
		public Float AvgTestLoss;
		public Float TestErrorPercentage;
		public Float Rate;
		public Float Momentum;
		public Float Beta2;
		public Float L2Penalty;
		public Float Dropout;
		public Float SampleRate;
		public DNNStates State;
		public DNNStates OldState;
		public DNNTaskStates TaskState;
		public Float fpropTime;
		public Float bpropTime;
		public Float updateTime;
		public bool PersistOptimizer;
        public bool DisableLocking;
        public bool PlainFormat;
        private bool disposedValue = false;

        public void OnElapsed(object? sender, System.Timers.ElapsedEventArgs e)
        {
            sb.Length = 0;
            if (Duration.Elapsed.Days > 0)
            {
                if (Duration.Elapsed.Days == 1)
                    sb.AppendFormat("{0:D} day {1:D2}:{2:D2}:{3:D2}", Duration.Elapsed.Days, Duration.Elapsed.Hours, Duration.Elapsed.Minutes, Duration.Elapsed.Seconds);
                else
                    sb.AppendFormat("{0:D} days {1:D2}:{2:D2}:{3:D2}", Duration.Elapsed.Days, Duration.Elapsed.Hours, Duration.Elapsed.Minutes, Duration.Elapsed.Seconds);
            }
            else
                sb.AppendFormat("{0:D2}:{1:D2}:{2:D2}", Duration.Elapsed.Hours, Duration.Elapsed.Minutes, Duration.Elapsed.Seconds);
            DurationString = sb.ToString();

            if (IsTraining)
            {
                var info = new TrainingInfo();
                DNNGetTrainingInfo(ref info);

                TotalCycles = info.TotalCycles;
                TotalEpochs = info.TotalEpochs;
                Cycle = info.Cycle;
                Epoch = info.Epoch; ;
                SampleIndex = info.SampleIndex;

                Rate = info.Rate;
                if (Optimizer != (DNNOptimizers)info.Optimizer)
                    Optimizer = (DNNOptimizers)info.Optimizer;

                Momentum = info.Momentum;
                Beta2 = info.Beta2;
                L2Penalty = info.L2Penalty;
                Gamma = info.Gamma;
                Dropout = info.Dropout;
                BatchSize = info.BatchSize;
                Height = info.Height;
                Width = info.Width;
                PadH = info.PadH;
                PadW = info.PadW;

                HorizontalFlip = info.HorizontalFlip;
                VerticalFlip = info.VerticalFlip;
                InputDropout = info.InputDropout;
                Cutout = info.Cutout;
                CutMix = info.CutMix;
                AutoAugment = info.AutoAugment;
                ColorCast = info.ColorCast;
                ColorAngle = info.ColorAngle;
                Distortion = info.Distortion;
                Interpolation = (DNNInterpolations)info.Interpolation;
                Scaling = info.Scaling;
                Rotation = info.Rotation;

                AvgTrainLoss = info.AvgTrainLoss;
                TrainErrorPercentage = info.TrainErrorPercentage;
                TrainErrors = info.TrainErrors;
                AvgTestLoss = info.AvgTestLoss;
                TestErrorPercentage = info.TestErrorPercentage;
                TestErrors = info.TestErrors;

                SampleRate = info.SampleSpeed;

                State = (DNNStates)info.State;
                TaskState = (DNNTaskStates)info.TaskState;

                AdjustedTrainSamplesCount = TrainingSamples % BatchSize == 0 ? TrainingSamples : ((TrainingSamples / BatchSize) + 1) * BatchSize;
                AdjustedTestSamplesCount = TestingSamples % BatchSize == 0 ? TestingSamples : ((TestingSamples / BatchSize) + 1) * BatchSize;

                TrainProgress(Optimizer, BatchSize, Cycle, TotalCycles, Epoch, TotalEpochs, HorizontalFlip, VerticalFlip, InputDropout, Cutout, CutMix, AutoAugment, ColorCast, ColorAngle, Distortion, Interpolation, Scaling, Rotation, SampleIndex, Rate, Momentum, Beta2, Gamma, L2Penalty, Dropout, AvgTrainLoss, TrainErrorPercentage, (Float)100 - TrainErrorPercentage, TrainErrors, AvgTestLoss, TestErrorPercentage, (Float)100 - TestErrorPercentage, TestErrors, State, TaskState);

                if (State != OldState)
                {
                    OldState = State;
                    SampleRate = (Float)0;
                }
            }
            else
            {
                var info = new TestingInfo();
                DNNGetTestingInfo(ref info);

                SampleIndex = info.SampleIndex;
                BatchSize = info.BatchSize;
                Height = info.Height;
                Width = info.Width;
                PadH = info.PadH;
                PadW = info.PadW;
                AvgTestLoss = info.AvgTestLoss;
                TestErrorPercentage = info.TestErrorPercentage;
                TestErrors = info.TestErrors;
                State = (DNNStates)(info.State);
                TaskState = (DNNTaskStates)info.TaskState;
                SampleRate = info.SampleSpeed;


                AdjustedTestSamplesCount = TestingSamples % BatchSize == 0 ? TestingSamples : ((TestingSamples / BatchSize) + 1) * BatchSize;

                TestProgress(BatchSize, SampleIndex, AvgTestLoss, TestErrorPercentage, (Float)100 - TestErrorPercentage, TestErrors, State, TaskState);

                if (State != OldState)
                {
                    OldState = State;
                    SampleRate = (Float)0;
                }
            }
        }

        static public string[] GetTextLabels(string fileName)
	    {	         
            int lines = 0;
            string[] list = new string[1];

            try
            {
                using (var streamReader = File.OpenText(fileName))
                {
                    var str = streamReader.ReadLine();
                    while (str != null)
                    {
                        lines++;
                        str = streamReader.ReadLine();
                    }
                }
               
                list = new string[lines];
                lines = 0;

                using (var streamReader = File.OpenText(fileName))
                {
                    var str = streamReader.ReadLine();
                    while (str != null)
                    {
                        list[lines++] = new string(str);
                        str = streamReader.ReadLine();
                    }
                }
            }
		    catch (IOException)
		    {
            }
		
            return list;
	    }

        static public ref DNNLayerInfo? GetLayerInfo(ref DNNLayerInfo? infoManaged, UInt layerIndex)
	    {
            if (infoManaged == null)
			    infoManaged = new DNNLayerInfo();

            var infoNative = new LayerInfo();
            DNNGetLayerInfo(layerIndex, ref infoNative);

            infoManaged.Name = (string)infoNative.Name;
            infoManaged.Description = (string)infoNative.Description;

            var layerType = (DNNLayerTypes)infoNative.LayerType;
            infoManaged.LayerType = layerType;
            infoManaged.IsNormLayer =
                layerType == DNNLayerTypes.BatchNorm ||
                layerType == DNNLayerTypes.BatchNormActivation ||
                layerType == DNNLayerTypes.BatchNormActivationDropout ||
                layerType == DNNLayerTypes.BatchNormRelu ||
                layerType == DNNLayerTypes.LayerNorm;

            infoManaged.Activation = (DNNActivations)infoNative.Activation;
            infoManaged.Algorithm = (DNNAlgorithms)infoNative.Algorithm;
            infoManaged.Cost = (DNNCosts)infoNative.Cost;
            infoManaged.NeuronCount = infoNative.NeuronCount;
            infoManaged.WeightCount = infoNative.WeightCount;
            infoManaged.BiasCount = infoNative.BiasesCount;
            infoManaged.LayerIndex = layerIndex; // infoNative.LayerIndex;
            infoManaged.InputCount = infoNative.InputsCount;
            UInt[] inputs = new UInt[infoNative.InputsCount];
            DNNGetLayerInputs(layerIndex, inputs);
            infoManaged.Inputs = [.. inputs];
            infoManaged.C = infoNative.C;
            infoManaged.D = infoNative.D;
            infoManaged.H = infoNative.H;
            infoManaged.W = infoNative.W;
            infoManaged.PadD = infoNative.PadD;
            infoManaged.PadH = infoNative.PadH;
            infoManaged.PadW = infoNative.PadW;
            infoManaged.KernelH = infoNative.KernelH;
            infoManaged.KernelW = infoNative.KernelW;
            infoManaged.StrideH = infoNative.StrideH;
            infoManaged.StrideW = infoNative.StrideW;
            infoManaged.DilationH = infoNative.DilationH;
            infoManaged.DilationW = infoNative.DilationW;
            infoManaged.Multiplier = infoNative.Multiplier;
            infoManaged.Groups = infoNative.Groups;
            infoManaged.Group = infoNative.Group;
            infoManaged.LocalSize = infoNative.LocalSize;
            infoManaged.Dropout = infoNative.Dropout;
            infoManaged.Weight = infoNative.Weight;
            infoManaged.GroupIndex = infoNative.GroupIndex;
            infoManaged.LabelIndex = infoNative.LabelIndex;
            infoManaged.InputC = infoNative.InputC;
            infoManaged.Alpha = infoNative.Alpha;
            infoManaged.Beta = infoNative.Beta;
            infoManaged.K = infoNative.K;
            infoManaged.FactorH = infoNative.fH;
            infoManaged.FactorW = infoNative.fW;
            infoManaged.HasBias = infoNative.HasBias;
            infoManaged.Scaling = infoManaged.IsNormLayer ? infoNative.Scaling : false;
            infoManaged.AcrossChannels = infoNative.AcrossChannels;
            infoManaged.LockUpdate = infoNative.Lockable ? (bool?)infoNative.Locked : (bool?)false;
            infoManaged.Lockable = infoNative.Lockable;

            //infoManaged.InputsNames = new System.Collections.Generic.List<string>();
            //foreach (var index in infoManaged.Inputs)
            //{
            //    var infoNativ = new LayerInfo();
            //    DNNGetLayerInfo(index, ref infoNativ);
            //    infoManaged.InputsNames.Add(infoNativ.Name);
            //}

            return ref infoManaged;
	    }

        public void ApplyParameters()
        {
            var info = new ModelInfo();
            DNNGetModelInfo(ref info);

            Name = new string(info.Name);
            Dataset = (DNNDatasets)info.Dataset;
            CostFunction = (DNNCosts)info.CostFunction;
            LayerCount = info.LayerCount;
            CostLayerCount = info.CostLayerCount;
            CostIndex = info.CostIndex;
            GroupIndex = info.GroupIndex;
            LabelIndex = info.LabelIndex;
            Hierarchies = info.Hierarchies;
            TrainingSamples = info.TrainSamplesCount;
            TestingSamples = info.TestSamplesCount;
            MeanStdNormalization = info.MeanStdNormalization;

            LabelsCollection = new string[Hierarchies][];

            switch (Dataset)
            {
                case DNNDatasets.tinyimagenet:
                    LabelsCollection[0] = GetTextLabels(Path.Combine(DatasetsDirectory, Dataset.ToString(), "classnames.txt"));
                    /*LabelsCollection[0] = new string[200];
                    for (int i = 0; i < 200; i++)
                        LabelsCollection[0][i] = i.ToString();*/
                    if (info.MeanTrainSet.Length >= 3)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            MeanTrainSet[i] = info.MeanTrainSet[i];
                            StdTrainSet[i] = info.StdTrainSet[i];
                        }
                    }
                    break;

                case DNNDatasets.cifar10:
                    LabelsCollection[0] = GetTextLabels(Path.Combine(DatasetsDirectory, Dataset.ToString(), "batches.meta.txt"));
                    if (info.MeanTrainSet.Length >= 3)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            MeanTrainSet[i] = info.MeanTrainSet[i];
                            StdTrainSet[i] = info.StdTrainSet[i];
                        }
                    }
                    break;

                case DNNDatasets.cifar100:
                    LabelsCollection[0] = GetTextLabels(Path.Combine(DatasetsDirectory, Dataset.ToString(), "coarse_label_names.txt"));
                    LabelsCollection[1] = GetTextLabels(Path.Combine(DatasetsDirectory, Dataset.ToString(), "fine_label_names.txt"));
                    if (info.MeanTrainSet.Length >= 3)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            MeanTrainSet[i] = info.MeanTrainSet[i];
                            StdTrainSet[i] = info.StdTrainSet[i];
                        }
                    }
                    break;

                case DNNDatasets.fashionmnist:
                    LabelsCollection[0] = GetTextLabels(Path.Combine(DatasetsDirectory, Dataset.ToString(), "batches.meta.txt"));
                    if (info.MeanTrainSet.Length >= 1)
                    {
                        for (int i = 0; i < 1; i++)
                        {
                            MeanTrainSet[i] = info.MeanTrainSet[i];
                            StdTrainSet[i] = info.StdTrainSet[i];
                        }
                    }
                    break;

                case DNNDatasets.mnist:
                    LabelsCollection[0] = new string[10];
                    for (int i = 0; i < 10; i++)
                        LabelsCollection[0][i] = i.ToString();
                    if (info.MeanTrainSet.Length >= 1)
                    {
                        for (int i = 0; i < 1; i++)
                        {
                            MeanTrainSet[i] = info.MeanTrainSet[i];
                            StdTrainSet[i] = info.StdTrainSet[i];
                        }
                    }
                    break;
            }

            Layers = new System.Collections.ObjectModel.ObservableCollection<DNNLayerInfo>();
            TrainingStrategies = new System.Collections.ObjectModel.ObservableCollection<DNNTrainingStrategy>();
            CostLayers = new DNNCostLayer[CostLayerCount];
                        
            UInt counter = 0;
            for (UInt layer = 0; layer < LayerCount; layer++)
            {
                DNNLayerInfo? inf = null;
                inf = GetLayerInfo(ref inf, layer);
                if (inf != null)
                {
                    Layers.Add(inf);

                    if (Layers[(int)layer].LayerType == DNNLayerTypes.Cost)
                        CostLayers[counter++] = new DNNCostLayer(Layers[(int)layer].Cost, Layers[(int)layer].LayerIndex, Layers[(int)layer].GroupIndex, Layers[(int)layer].LabelIndex, Layers[(int)layer].NeuronCount, Layers[(int)layer].Name, Layers[(int)layer].Weight);
                }
            }

            GroupIndex = CostLayers[CostIndex].GroupIndex;
            LabelIndex = CostLayers[CostIndex].LabelIndex;
            ClassCount = CostLayers[CostIndex].ClassCount;

            Optimizer = (DNNOptimizers)GetOptimizer();
        }

        public DNNModel(string definition)
        {
		    Duration = new System.Diagnostics.Stopwatch();
            sb = new System.Text.StringBuilder();
            State = DNNStates.Idle;
		    OldState = DNNStates.Idle;
		    TaskState = DNNTaskStates.Stopped;
            MeanTrainSet = new Float[] { (Float)0, (Float)0, (Float)0 };
            StdTrainSet = new Float[] { (Float)0, (Float)0, (Float)0 };
            StorageDirectory = Path.Combine(Environment.GetFolderPath(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Environment.SpecialFolder.MyDocuments : Environment.SpecialFolder.UserProfile), "convnet");
		    DatasetsDirectory = Path.Combine(StorageDirectory, "datasets");
		    DefinitionsDirectory = Path.Combine(StorageDirectory, "definitions");

	        GroupIndex = 0;
		    LabelIndex = 0;
		    CostIndex = 0;
		    Multiplier = 1;

		    DNNDataprovider(StorageDirectory);

            CheckMsg checkMsg = new CheckMsg();
		    if (DNNRead(definition, ref checkMsg) == 1)
		    {
			    DNNLoadDataset();
                Definition = definition;
                ApplyParameters();

                WorkerTimer = new System.Timers.Timer(1000.0);
                WorkerTimer.Elapsed += new System.Timers.ElapsedEventHandler(OnElapsed);
            }
		    else
		    {
			    throw new Exception(checkMsg.Message);
		    }
	    }

        public void SetPersistOptimizer(bool persist)
        {
            DNNPersistOptimizer(persist);
            PersistOptimizer = persist;
        }

        public void SetDisableLocking(bool disable)
        {
            DNNDisableLocking(disable);
            DisableLocking = disable;
        }

        public void GetConfusionMatrix()
        {
            var classCount = CostLayers[CostIndex].ClassCount;
            ConfusionMatrix = new UInt[classCount * classCount];
            DNNGetConfusionMatrix(CostIndex, ConfusionMatrix);
        }

        public bool SetShuffleCount(UInt count)
        {
            return DNNSetShuffleCount(count);
        }

        public bool BatchNormUsed()
        {
            return DNNBatchNormUsed();
        }

        public static bool StochasticEnabled()
        {
            return DNNStochasticEnabled();
        }

        public void UpdateLayerStatistics(ref DNNLayerInfo? info, UInt layerIndex, bool updateUI)
        {
            if (info != null)
            {
                var statsInfo = new StatsInfo();
                DNNRefreshStatistics(layerIndex, ref statsInfo);

                info.Description = statsInfo.Description;
                info.NeuronsStats = new DNNStats(ref statsInfo.NeuronsStats);
                info.WeightsStats = new DNNStats(ref statsInfo.WeightsStats);
                info.BiasesStats = new DNNStats(ref statsInfo.BiasesStats);
                info.FPropLayerTime = statsInfo.FPropLayerTime;
                info.BPropLayerTime = statsInfo.BPropLayerTime;
                info.UpdateLayerTime = statsInfo.UpdateLayerTime;
                fpropTime = statsInfo.FPropTime;
                bpropTime = statsInfo.BPropTime;
                updateTime = statsInfo.UpdateTime;
                info.LockUpdate = info.Lockable ? (bool?)statsInfo.Locked : (bool?)false;

                if (updateUI)
                {
                    switch (info.LayerType)
                    {
                        case DNNLayerTypes.Input:
                            {
                                var channels = info.C;
                                var color = channels == 3;
                                var width = info.W;
                                var height = info.H;
                                var area = width * height;
                                var nativeTotalSize = color ? 3 * area : area + width;
                                var totalSize = 4 * area;
                                var format = Avalonia.Platform.PixelFormat.Rgba8888;
                                var stride = (int)width * ((format.BitsPerPixel + 7) / 8);
                                var snapshot = new Float[nativeTotalSize];
                                var labelVector = new UInt64[Hierarchies];
                                
                                if (totalSize > 0 && totalSize <= int.MaxValue)
                                {
                                    var pictureLoaded = DNNGetInputSnapShot(snapshot, labelVector);

                                    if (pictureLoaded)
                                    {
                                        var img = new Byte[nativeTotalSize];
                                        if (MeanStdNormalization)
                                            for (UInt channel = 0; channel < channels; channel++)
                                                for (UInt hw = 0; hw < area; hw++)
                                                    img[(hw * channels) + channel] = pictureLoaded ? FloatSaturate((snapshot[hw + channel * area] * StdTrainSet[channel]) + MeanTrainSet[channel]) : FloatSaturate(MeanTrainSet[channel]);
                                        else
                                            for (UInt channel = 0; channel < channels; channel++)
                                                for (UInt hw = 0; hw < area; hw++)
                                                    img[(hw * channels) + channel] = pictureLoaded ? FloatSaturate((snapshot[hw + channel * area] + (Float)(2)) * 64) : FloatSaturate(128);

                                        var newImg = new Byte[totalSize];
                                        if (color)
                                        {
                                            for (var i = 0ul; i < area; i++)
                                            {
                                                newImg[(i * 4) + 0] = img[(i * 3) + 0];  // R
                                                newImg[(i * 4) + 1] = img[(i * 3) + 1];  // G
                                                newImg[(i * 4) + 2] = img[(i * 3) + 2];  // B
                                                newImg[(i * 4) + 3] = 255;               // A
                                            }
                                        }
                                        else
                                        {
                                            for (var i = 0ul; i < area; i++)
                                            {
                                                newImg[(i * 4) + 0] = img[i];  // R
                                                newImg[(i * 4) + 1] = img[i];  // G
                                                newImg[(i * 4) + 2] = img[i];  // B
                                                newImg[(i * 4) + 3] = 255;     // A
                                            }
                                        }

                                        int length = Marshal.SizeOf(newImg[0]) * newImg.Length;
                                        IntPtr pnt = Marshal.AllocHGlobal(length);
                                        Marshal.Copy(newImg, 0, pnt, newImg.Length);
                                        var bitmap = new WriteableBitmap(format, AlphaFormat.Premul, pnt, new PixelSize((int)width, (int)height), new Vector(96, 96), stride);
                                        Marshal.FreeHGlobal(pnt);
                                        InputSnapshot = bitmap;
                                        Label = LabelsCollection[LabelIndex][labelVector[LabelIndex]];
                                        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
                                    }
                                    else
                                        Label = System.String.Empty;
                                }
                            }
                            break;

                        case DNNLayerTypes.Convolution:
                        case DNNLayerTypes.ConvolutionTranspose:
                        case DNNLayerTypes.DepthwiseConvolution:
                        case DNNLayerTypes.PartialDepthwiseConvolution:
                            {
                                var depthwise = info.LayerType == DNNLayerTypes.DepthwiseConvolution || info.LayerType == DNNLayerTypes.PartialDepthwiseConvolution;
                                var color = !depthwise && info.InputC == 3;
                                var border = (info.InputC != 3 && info.KernelH == 1 && info.KernelW == 1) ? (UInt)0 : (UInt)1;
                                var pitchH = info.KernelH + border;
                                var pitchW = info.KernelW + border;
                                var width = info.C * pitchH + border;
                                var height = (info.InputC == 3) ? (pitchW + 3 * border) : (depthwise ? (pitchW + border) : ((info.InputC / info.Groups) * pitchW + border));
                                var area = height * width;
                                var nativeTotalSize = color ? 3 * area : area + width;
                                var totalSize = 4 * area;
                                var format = Avalonia.Platform.PixelFormat.Rgba8888;
                                var stride = (int)width * ((format.BitsPerPixel + 7) / 8);

                                if (totalSize > 0 && totalSize <= int.MaxValue)
                                {
                                    var img = new Byte[nativeTotalSize];
                                    DNNGetImage(layerIndex, BackgroundColor, img);
                                    
                                    var newImg = new Byte[totalSize];
                                    if (color)
                                    {
                                        for (var i = 0ul; i < area; i++)
                                        {
                                            newImg[(i * 4) + 0] = img[(i * 3) + 0];  // R
                                            newImg[(i * 4) + 1] = img[(i * 3) + 1];  // G
                                            newImg[(i * 4) + 2] = img[(i * 3) + 2];  // B
                                            newImg[(i * 4) + 3] = 255;               // A
                                        }
                                    }
                                    else
                                    {
                                        for (var i = 0ul; i < area; i++)
                                        {
                                            newImg[(i * 4) + 0] = img[i];  // R
                                            newImg[(i * 4) + 1] = img[i];  // G
                                            newImg[(i * 4) + 2] = img[i];  // B
                                            newImg[(i * 4) + 3] = 255;     // A
                                        }
                                    }
                                    
                                    int length = Marshal.SizeOf(newImg[0]) * newImg.Length;
                                    IntPtr pnt = Marshal.AllocHGlobal(length);
                                    Marshal.Copy(newImg, 0, pnt, newImg.Length);
                                    var bitmap = new WriteableBitmap(format, AlphaFormat.Premul, pnt, new PixelSize((int)width, (int)height), new Vector(96, 96), stride);
                                    Marshal.FreeHGlobal(pnt);
                                    info.WeightsSnapshotX = (int)(width * BlockSize);
                                    info.WeightsSnapshotY = (int)(height * BlockSize);
                                    info.WeightsSnapshot = bitmap;
                                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
                                }
                            }
                            break;

                        case DNNLayerTypes.BatchNorm:
                        case DNNLayerTypes.BatchNormActivation:
                        case DNNLayerTypes.BatchNormActivationDropout:
                        case DNNLayerTypes.BatchNormRelu:
                        case DNNLayerTypes.Dense:
                        case DNNLayerTypes.GroupNorm:
                        case DNNLayerTypes.LayerNorm:
                            {
                                if (info.BiasCount > 0)
                                {
                                    var width = info.BiasCount;
                                    var height = (info.WeightCount / width) + 3;
                                    var area = width * height;
                                    var nativeTotalSize = area + width;
                                    var totalSize = 4 * area;
                                    var format = Avalonia.Platform.PixelFormat.Rgba8888;
                                    var stride = (int)width * ((format.BitsPerPixel + 7) / 8);

                                    if (totalSize > 0 && totalSize <= int.MaxValue)
                                    {
                                        var img = new Byte[(int)(nativeTotalSize)];
                                        DNNGetImage(info.LayerIndex, BackgroundColor, img);

                                        var newImg = new Byte[totalSize];
                                        for (var i = 0ul; i < area; i++)
                                        {
                                            newImg[(i * 4) + 0] = img[i];  // R
                                            newImg[(i * 4) + 1] = img[i];  // G
                                            newImg[(i * 4) + 2] = img[i];  // B
                                            newImg[(i * 4) + 3] = 255;     // A
                                        }
                                        
                                        int length = Marshal.SizeOf(newImg[0]) * newImg.Length;
                                        IntPtr pnt = Marshal.AllocHGlobal(length);
                                        Marshal.Copy(newImg, 0, pnt, newImg.Length);
                                        var bitmap = new WriteableBitmap(format, AlphaFormat.Premul, pnt, new PixelSize((int)width, (int)height), new Vector(96, 96), stride);
                                        Marshal.FreeHGlobal(pnt);
                                        info.WeightsSnapshotX = (int)(width * BlockSize);
                                        info.WeightsSnapshotY = (int)(height * BlockSize);
                                        info.WeightsSnapshot = bitmap;
                                        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
                                    }
                                }
                            }
                            break;

                        case DNNLayerTypes.PRelu:
                            {
                                var width = info.WeightCount;
                                var height = (UInt)4;
                                var area = width * height;
                                var nativeTotalSize = area;
                                var totalSize = 4 * area;
                                var format = Avalonia.Platform.PixelFormat.Rgba8888;
                                var stride = (int)width * ((format.BitsPerPixel + 7) / 8);

                                if (totalSize > 0 && totalSize <= int.MaxValue)
                                {
                                    var img = new Byte[(int)(nativeTotalSize)];
                                    DNNGetImage(info.LayerIndex, BackgroundColor, img);

                                    var newImg = new Byte[totalSize];
                                    for (var i = 0ul; i < area; i++)
                                    {
                                        newImg[(i * 4) + 0] = img[i];  // R
                                        newImg[(i * 4) + 1] = img[i];  // G
                                        newImg[(i * 4) + 2] = img[i];  // B
                                        newImg[(i * 4) + 3] = 255;     // A
                                    }

                                    int length = Marshal.SizeOf(newImg[0]) * newImg.Length;
                                    IntPtr pnt = Marshal.AllocHGlobal(length);
                                    Marshal.Copy(newImg, 0, pnt, newImg.Length);
                                    var bitmap = new WriteableBitmap(format, AlphaFormat.Premul, pnt, new PixelSize((int)width, (int)height), new Vector(96, 96), stride);
                                    Marshal.FreeHGlobal(pnt);
                                    info.WeightsSnapshotX = (int)(width * BlockSize);
                                    info.WeightsSnapshotY = (int)(height * BlockSize);
                                    info.WeightsSnapshot = bitmap;
                                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
                                }
                            }
                            break;
                    }
                }
            }
        }

        public void UpdateLayerInfo(UInt layerIndex, bool updateUI)
        {
           var layer = Layers[(int)layerIndex];

            if (layerIndex == 0)
                GetLayerInfo(ref layer, layerIndex);

            UpdateLayerStatistics(ref layer, layerIndex, updateUI);
        }

        public bool SetFormat(bool plain)
        {
            var ret = DNNSetFormat(plain);

            if (ret)
                PlainFormat = plain;

            return ret;
        }

        public void SetOptimizer(DNNOptimizers strategy)
        {
            if (strategy != Optimizer)
            {
                DNNSetOptimizer((Optimizers)strategy);
                Optimizer = strategy;
            }
        }

        public void ResetOptimizer()
        {
            DNNResetOptimizer();
        }

        public void SetUseTrainingStrategy(bool enable)
        {
            DNNSetUseTrainingStrategy(enable);
            UseTrainingStrategy = enable;
        }

        public void SetCostIndex(UInt index)
        {
            DNNSetCostIndex(index);

            CostIndex = index;
            GroupIndex = CostLayers[CostIndex].GroupIndex;
            LabelIndex = CostLayers[CostIndex].LabelIndex;
            ClassCount = CostLayers[CostIndex].ClassCount;
        }

        public void UpdateCostInfo(UInt index)
        {
            var info = new CostInfo();
            DNNGetCostInfo(index, ref info);

            CostLayers[index].TrainErrors = info.TrainErrors;
            CostLayers[index].TrainLoss = info.TrainLoss;
            CostLayers[index].AvgTrainLoss = info.AvgTrainLoss;
            CostLayers[index].TrainErrorPercentage = info.TrainErrorPercentage;
            CostLayers[index].TrainAccuracy = (Float)100 - info.TrainErrorPercentage;

            CostLayers[index].TestErrors = info.TestErrors;
            CostLayers[index].TestLoss = info.TestLoss;
            CostLayers[index].AvgTestLoss = info.AvgTestLoss;
            CostLayers[index].TestErrorPercentage = info.TestErrorPercentage;
            CostLayers[index].TestAccuracy = (Float)100 - info.TestErrorPercentage;
        }

        public void AddTrainingRate(DNNTrainingRate rate, bool clear, UInt gotoEpoch, UInt trainSamples)
        {
            var nativeRate = new TrainingRate((Optimizers)rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, rate.Cycles, rate.Epochs, rate.EpochMultiplier, rate.MaximumRate, rate.MinimumRate, rate.FinalRate, rate.Gamma, rate.DecayAfterEpochs, rate.DecayFactor, rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, (Interpolations)rate.Interpolation, rate.Scaling, rate.Rotation);

            DNNAddTrainingRate(ref nativeRate, clear, gotoEpoch, trainSamples);
        }

        public void AddTrainingRateSGDR(DNNTrainingRate rate, bool clear, UInt gotoEpoch, UInt gotoCycle, UInt trainSamples)
	    {
		    var nativeRate = new TrainingRate((Optimizers)rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, rate.Cycles, rate.Epochs, rate.EpochMultiplier, rate.MaximumRate, rate.MinimumRate, rate.FinalRate, rate.Gamma, rate.DecayAfterEpochs, rate.DecayFactor, rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, (Interpolations)rate.Interpolation, rate.Scaling, rate.Rotation);

            DNNAddTrainingRateSGDR(ref nativeRate, clear, gotoEpoch, gotoCycle, trainSamples);
        }

        public void ClearTrainingStrategies()
        {
            DNNClearTrainingStrategies();
        }

        public void AddTrainingStrategy(DNNTrainingStrategy strategy)
        {
            var nativeStrategy = new TrainingStrategy(strategy.Epochs, strategy.N, strategy.D, strategy.H, strategy.W, strategy.PadD, strategy.PadH, strategy.PadW, strategy.Momentum, strategy.Beta2, strategy.Gamma, strategy.L2Penalty, strategy.Dropout, strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, (Interpolations)strategy.Interpolation, strategy.Scaling, strategy.Rotation);

            DNNAddTrainingStrategy(ref nativeStrategy);
        }

        public void Start(bool training)
        {
            unsafe
            {
                if (NewEpoch != null)
                    DNNSetNewEpochDelegate(Marshal.GetFunctionPointerForDelegate(NewEpoch).ToPointer());
            }          

            SampleRate = (Float)0;
            State = DNNStates.Idle;
            IsTraining = training;
            if (IsTraining)
                DNNTraining();
            else
                DNNTesting();

            TaskState = DNNTaskStates.Running;
            WorkerTimer.Start();
            Duration.Start();
        }

        public void Stop()
        {
            SampleRate = (Float)0;
            Duration.Reset();
            DNNStop();
            WorkerTimer.Stop();
            State = DNNStates.Completed;
            TaskState = DNNTaskStates.Stopped;
        }

        public void Pause()
        {
            WorkerTimer.Stop();
            Duration.Stop();
            DNNPause();
            TaskState = DNNTaskStates.Paused;
        }

        public void Resume()
        {
            DNNResume();
            Duration.Start();
            WorkerTimer.Start();
            TaskState = DNNTaskStates.Running;
        }

        public void SetLocked(bool locked)
        {
            DNNSetLocked(locked);
            for (var i = (UInt)0; i < LayerCount; i++)
                if (Layers[(int)i].Lockable)
                    Layers[(int)i].LockUpdate = locked;
        }

        public void SetLayerLocked(UInt layerIndex, bool locked)
        {
            DNNSetLayerLocked(layerIndex, locked);
        }

        public DNNCheckMsg Check(ref StringBuilder definition)
	    {
		    var checkMsg = new CheckMsg();

            DNNCheck(definition, ref checkMsg);

            return new DNNCheckMsg(checkMsg.Row, checkMsg.Column, checkMsg.Message, checkMsg.Error, definition.ToString());
        }
            
        public int Load(string fileName)
        {
            var checkMsg = new CheckMsg();

            DNNModelDispose();
            DNNDataprovider(StorageDirectory);

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);

            if (DNNLoad(fileName, ref checkMsg) == 1)
            {
                DNNLoadDataset();

                var reader = new System.IO.StreamReader(fileName, true);
                Definition = reader.ReadToEnd();
                reader.Close();

                DNNResetWeights();
                ApplyParameters();
            }
            else
                throw new Exception(checkMsg.Message);

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);

            return 1;
        }

        public bool LoadDataset()
        {
            return DNNLoadDataset();
        }

        public bool LoadModel(string fileName)
        {
            return DNNLoadModel(fileName);
        }

        public bool SaveModel(string fileName)
        {
            return DNNSaveModel(fileName);
        }

        public bool ClearLog()
        {
            return DNNClearLog();
        }

        public bool LoadLog(string fileName)
        {
            return DNNLoadLog(fileName);
        }

        public bool SaveLog(string fileName)
        {
            return DNNSaveLog(fileName);
        }

        public int LoadWeights(string fileName, bool persist)
        {
            var ret = DNNLoadWeights(fileName, persist);

            Optimizer = (DNNOptimizers)GetOptimizer();

            if (ret == 0 && SelectedIndex > 0)
            {
                var layerInfo = Layers[SelectedIndex];
                UpdateLayerStatistics(ref layerInfo, (UInt)SelectedIndex, true);
            }

            return ret;
        }

        public int SaveWeights(string fileName, bool persist)
        {
            return DNNSaveWeights(fileName, persist);
        }

        public void ResetWeights()
        {
            DNNResetWeights();
        }

        public int LoadLayerWeights(string fileName, UInt layerIndex)
        {
            var ret = DNNLoadLayerWeights(fileName, layerIndex, false);

            if (ret == 0 && SelectedIndex > 0)
            {
                var layerInfo = Layers[(int)layerIndex];
                UpdateLayerStatistics(ref layerInfo, layerIndex, layerIndex == (UInt)SelectedIndex);
            }

            return ret;
        }

        public int SaveLayerWeights(string fileName, UInt layerIndex)
        {
            return DNNSaveLayerWeights(fileName, layerIndex, false);
        }

        public void ResetLayerWeights(UInt layerIndex)
        {
            DNNResetLayerWeights(layerIndex);
        }
      
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                    WorkerTimer.Close();
                }

                DNNModelDispose();
                DNNDataproviderDispose();
                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~DNNModel()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    };
}