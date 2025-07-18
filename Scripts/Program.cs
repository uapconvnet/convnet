﻿using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.InteropServices;

using Float = System.Single;
using UInt = System.UInt64;


namespace Scripts
{
    [Serializable()]
    public enum Scripts
    {
        augshufflenet = 0,
        densenet = 1,
        efficientnetv2 = 2,
        mobilenetv3 = 3,
        resnet = 4,
        shufflenetv2 = 5
    }

    [Serializable()]
    public enum Datasets
    {
        cifar10 = 0,
        cifar100 = 1,
        fashionmnist = 2,
        mnist = 3,
        tinyimagenet = 4
    }

    [Serializable()]
    public enum Fillers
    {
        Constant = 0,
        HeNormal = 1,
        HeUniform = 2,
        LeCunNormal = 3,
        LeCunUniform = 4,
        Normal = 5,
        TruncatedNormal = 6,
        Uniform = 7,
        XavierNormal = 8,
        XavierUniform = 9
    }

    [Serializable()]
    public enum FillerModes
    {
        Avg = 0,
        In = 1,
        Out = 2
    }

    [Serializable()]
    public enum Activations
    {
        FRelu = 1,
        HardSwish = 10,
        HardSigmoid = 11,
        Sigmoid = 12,
        Mish = 16,
        Relu = 19,
        Swish = 25,
        TanhExp = 27,
        Gelu = 28
    }

    [Serializable()]
    public class EfficientNetRecord(UInt expandRatio = 4, UInt channels = 24, UInt iterations = 2, UInt stride = 1, bool se = false)
    {
        public UInt ExpandRatio { get; set; } = expandRatio;
        public UInt Channels { get; set; } = channels;
        public UInt Iterations { get; set; } = iterations;
        public UInt Stride { get; set; } = stride;
        public bool SE { get; set; } = se;
        
        public override string ToString()
        {
            return "(" + ExpandRatio.ToString() + "-" + Channels.ToString() + "-" + Iterations.ToString() + "-" + Stride.ToString() + (SE ? "-se" : "") + ")";
        }
    }

    [Serializable()]
    public class ShuffleNetRecord(UInt iterations = 6u, UInt kernel = 3u, UInt pad = 1u, UInt shuffle = 2u, bool se = false)
    {
	public UInt Iterations { get; set; } = iterations;
        public UInt Kernel { get; set; } = kernel;
        public UInt Pad { get; set; } = pad;
        public UInt Shuffle { get; set; } = shuffle;
        public bool SE { get; set; } = se;
        
        public override string ToString()
        {
            return "(" + Iterations.ToString() + "-" + Kernel.ToString() + "-" + Pad.ToString() + "-" + Shuffle.ToString() + (SE ? "-se" : "") + ")";
        }
    }

    [Serializable()]
    public class ScriptParameters(Scripts script = Scripts.shufflenetv2, Datasets dataset = Datasets.cifar10, UInt h = 32, UInt w = 32, UInt padH = 4, UInt padW = 4, bool mirrorPad = false, bool meanStdNorm = true, Fillers weightsFiller = Fillers.HeNormal, FillerModes weightsFillerMode = FillerModes.In, Float weightsGain = (Float)1.0, Float weightsScale = (Float)0.05, Float weightsLRM = 1, Float weightsWDM = 1, bool hasBias = false, Fillers biasesFiller = Fillers.Constant, FillerModes biasesFillerMode = FillerModes.In, Float biasesGain = (Float)1.0, Float biasesScale = 0, Float biasesLRM = 1, Float biasesWDM = 1, Float batchNormMomentum = (Float)0.995, Float batchNormEps = (Float)1E-04, bool batchNormScaling = false, Float alpha = (Float)0, Float beta = (Float)0, UInt groups = 3, UInt iterations = 4, UInt width = 8, UInt growthRate = 12, bool bottleneck = false, Float dropout = 0, Float compression = 0, bool squeezeExcitation = false, bool channelZeroPad = true, Activations activation = Activations.Relu, UInt strideHFirstConv = 2, UInt strideWFirstConv = 2, Float depthDrop = (Float)0.2, bool fixedDepthDrop = false)
    {
        public Scripts Script { get; set; } = script;

        public Datasets Dataset { get; set; } = dataset;

        public UInt C { get; set; } = 3u;

        public UInt D { get; set; } = 1u;

        public UInt H { get; set; } = h;

        public UInt W { get; set; } = w;

        public UInt PadD { get; set; } = 0u;

        public UInt PadH { get; set; } = padH;

        public UInt PadW { get; set; } = padW;

        public bool MirrorPad { get; set; } = mirrorPad;

        public bool MeanStdNormalization { get; set; } = meanStdNorm;

        public Fillers WeightsFiller { get; set; } = weightsFiller;

        public FillerModes WeightsFillerMode { get; set; } = weightsFillerMode;

        public Float WeightsGain { get; set; } = weightsGain;

        public Float WeightsScale { get; set; } = weightsScale;

        public Float WeightsLRM { get; set; } = weightsLRM;

        public Float WeightsWDM { get; set; } = weightsWDM;

        public bool HasBias { get; set; } = hasBias;

        public Fillers BiasesFiller { get; set; } = biasesFiller;

        public FillerModes BiasesFillerMode { get; set; } = biasesFillerMode;

        public Float BiasesGain { get; set; } = biasesGain;

        public Float BiasesScale { get; set; } = biasesScale;

        public Float BiasesLRM { get; set; } = biasesLRM;

        public Float BiasesWDM { get; set; } = biasesWDM;

        public bool FixedDepthDrop { get; set; } = fixedDepthDrop;

        public Float DepthDrop { get; set; } = depthDrop;

        public Float BatchNormMomentum { get; set; } = batchNormMomentum;

        public Float BatchNormEps { get; set; } = batchNormEps;

        public bool BatchNormScaling { get; set; } = batchNormScaling;

        public Float Alpha { get; set; } = alpha;

        public Float Beta { get; set; } = beta;

        public UInt Groups { get; set; } = groups;

        public UInt Iterations { get; set; } = iterations;

        public UInt Width { get; set; } = width;

        public UInt GrowthRate { get; set; } = growthRate;

        public bool Bottleneck { get; set; } = bottleneck;

        public Float Dropout { get; set; } = dropout;

        public Float Compression { get; set; } = compression;

        public bool SqueezeExcitation { get; set; } = squeezeExcitation;

        public bool ChannelZeroPad { get; set; } = channelZeroPad;

        public Activations Activation { get; set; } = activation;

        public UInt StrideHFirstConv { get; set; } = strideHFirstConv;

        public UInt StrideWFirstConv { get; set; } = strideWFirstConv;

        public ObservableCollection<EfficientNetRecord> EfficientNet { get; set; } = [new(1, 24, 2, 1, false), new(4, 48, 4, 2, false), new(4, 64, 4, 2, false), new(4, 128, 6, 2, true), new(6, 160, 9, 1, true), new(6, 256, 15, 2, true)];

        public ObservableCollection<ShuffleNetRecord> ShuffleNet { get; set; } = [new(7, 3, 1, 2, false), new(7, 3, 1, 2, true), new(7, 3, 1, 2, true)];

        public bool RandomCrop { get { return PadH > 0 || PadW > 0; } }

        public IEnumerable<Scripts> ScriptsList { get { return Enum.GetValues(typeof(Scripts)).Cast<Scripts>(); } }

        public IEnumerable<Datasets> DatasetsList { get { return Enum.GetValues(typeof(Datasets)).Cast<Datasets>(); } }

        public IEnumerable<Activations> ActivationsList { get { return Enum.GetValues(typeof(Activations)).Cast<Activations>(); } }

        public IEnumerable<Fillers> FillersList { get { return Enum.GetValues(typeof(Fillers)).Cast<Fillers>(); } }

        public IEnumerable<FillerModes> FillerModesList { get { return Enum.GetValues(typeof(FillerModes)).Cast<FillerModes>(); } }

        public bool WeightsFillerModeVisible { get { return WeightsFiller == Fillers.HeNormal || WeightsFiller == Fillers.HeUniform || WeightsFiller == Fillers.LeCunNormal || WeightsFiller == Fillers.LeCunUniform; } }

        public bool WeightsGainVisible { get { return WeightsFiller == Fillers.HeNormal || WeightsFiller == Fillers.HeUniform || WeightsFiller == Fillers.LeCunNormal || WeightsFiller == Fillers.LeCunUniform || WeightsFiller == Fillers.XavierNormal || WeightsFiller == Fillers.XavierUniform; } }

        public bool WeightsScaleVisible { get { return WeightsFiller == Fillers.Constant || WeightsFiller == Fillers.Normal || WeightsFiller == Fillers.TruncatedNormal || WeightsFiller == Fillers.Uniform; } }

        public bool BiasesFillerModeVisible { get { return BiasesFiller == Fillers.HeNormal || BiasesFiller == Fillers.HeUniform || BiasesFiller == Fillers.LeCunNormal || BiasesFiller == Fillers.LeCunUniform; } }

        public bool BiasesGainVisible { get { return BiasesFiller == Fillers.HeNormal || BiasesFiller == Fillers.HeUniform || BiasesFiller == Fillers.LeCunNormal || BiasesFiller == Fillers.LeCunUniform || BiasesFiller == Fillers.XavierNormal || BiasesFiller == Fillers.XavierUniform; } }

        public bool BiasesScaleVisible { get { return BiasesFiller == Fillers.Constant || BiasesFiller == Fillers.Normal || BiasesFiller == Fillers.TruncatedNormal || BiasesFiller == Fillers.Uniform; } }

        public bool DropoutUsed { get { return (Dropout > 0 && Dropout < 1); } }

        public bool GroupsVisible { get { return Script != Scripts.efficientnetv2 && Script != Scripts.shufflenetv2 && Script != Scripts.augshufflenet; } }

        public bool IterationsVisible { get { return Script != Scripts.efficientnetv2 && Script != Scripts.shufflenetv2 && Script != Scripts.augshufflenet; } }

        public bool WidthVisible { get { return Script == Scripts.mobilenetv3 || Script == Scripts.resnet || Script == Scripts.shufflenetv2 || Script == Scripts.augshufflenet; } }

        public bool GrowthRateVisible { get { return Script == Scripts.densenet; } }

        public bool DropoutVisible { get { return Script == Scripts.densenet || Script == Scripts.resnet || Script == Scripts.efficientnetv2; } }

        public bool DepthDropVisible { get { return Script == Scripts.efficientnetv2 || Script == Scripts.mobilenetv3 || Script == Scripts.resnet || Script == Scripts.densenet; } }

        public bool CompressionVisible { get { return Script == Scripts.densenet; } }

        public bool BottleneckVisible { get { return Script == Scripts.densenet || Script == Scripts.resnet; } }

        public bool SqueezeExcitationVisible { get { return Script == Scripts.mobilenetv3; } }

        public bool ChannelZeroPadVisible { get { return Script == Scripts.resnet; } }

        public bool EfficientNetVisible { get { return Script == Scripts.efficientnetv2; } }

        public bool ShuffleNetVisible { get { return Script == Scripts.shufflenetv2 || Script == Scripts.augshufflenet; } }
        
        public UInt Depth
        {
            get
            {
                switch (Script)
                {
                    case Scripts.densenet:
                        return (Groups * Iterations * (Bottleneck ? 2u : 1u)) + ((Groups - 1) * 2);
                    case Scripts.mobilenetv3:
                        return (Groups * Iterations * 3) + ((Groups - 1) * 2);
                    case Scripts.resnet:
                        return (Groups * Iterations * (Bottleneck ? 3u : 2u)) + ((Groups - 1) * 2);
                    default:
                        return 0;
                }
            }
        }
        
        public string ModelName
        {
            get
            {
                switch (Script)
                {
                    case Scripts.densenet:
                        return Script.ToString() + "-" + Groups.ToString() + "-" + Iterations.ToString() + "-" + GrowthRate.ToString() + (Dropout > 0 ? "-dropout" : "") + (DepthDrop > 0 ? (FixedDepthDrop ? "-fixeddepthdrop" : "-depthdrop") : "") + (Compression > 0 ? "-compression" : "") + (Bottleneck ? "-bottleneck" : "") + "-" + Activation.ToString().ToLower();
                    case Scripts.efficientnetv2:
                        {
                            string name = "";
                            foreach (var rec in EfficientNet)
                                name += rec.ToString();
                            return Script.ToString() + (DepthDrop > 0 ? (FixedDepthDrop ? "-fixeddepthdrop-" : "-depthdrop-") : "") + name;
                        }
                    case Scripts.mobilenetv3:
                        return Script.ToString() + "-" + Groups.ToString() + "-" + Iterations.ToString() + "-" + Width.ToString() + "-" + Activation.ToString().ToLower() + (SqueezeExcitation ? " -se" : "") + (DepthDrop > 0 ? (FixedDepthDrop ? "-fixeddepthdrop" : "-depthdrop") : "");
                    case Scripts.resnet:
                        return Script.ToString() + "-" + Groups.ToString() + "-" + Iterations.ToString() + "-" + Width.ToString() + (Dropout > 0 ? "-dropout" : "") + (DepthDrop > 0 ? (FixedDepthDrop ? "-fixeddepthdrop" : "-depthdrop") : "") + (Bottleneck ? "-bottleneck" : "") + (ChannelZeroPad ? "-channelzeropad" : "") + "-" + Activation.ToString().ToLower();
                    case Scripts.augshufflenet:
                    case Scripts.shufflenetv2:
                        {
                            string name = "";
                            foreach (var rec in ShuffleNet)
                                name += rec.ToString();
                            return Script.ToString() + "-" + Width.ToString() + name;
                        }
                    default:
                        return Script.ToString() + "-" + Groups.ToString() + "-" + Iterations.ToString();
                }
            }
        }
        
        public UInt Classes
        {
            get
            {
                switch (Dataset)
                {
                    case Datasets.cifar100:
                        return 100;
                    case Datasets.tinyimagenet:
                        return 200;
                    default:
                        return 10;
                }
            }
        }
    }

    public class ScriptCatalog
    {
        public static string nwl { get; } = Environment.NewLine;


        public static string to_string(bool variable)
        {
            return variable ? "Yes" : "No";
        }

        public static string to_string(UInt number)
        {
            return number.ToString();
        }

        public static string to_string(Float number)
        {
            return number.ToString(new CultureInfo("en-US"));
        }

        public static string to_string(Datasets dataset)
        {
            return dataset.ToString();
        }

        public static string to_string(Fillers filler)
        {
            return filler.ToString();
        }

        public static string to_string(FillerModes fillerMode)
        {
            return fillerMode.ToString();
        }

        public static UInt DIV8(UInt channels)
        {
            if (channels % 8ul == 0ul)
                return channels;

            return ((channels / 8ul) + 1ul) * 8ul;
        }

        public static UInt DIV16(UInt channels)
        {
            if (channels % 16ul == 0ul)
                return channels;

            return ((channels / 16ul) + 1ul) * 16ul;
        }

        public static string In(string prefix, UInt id)
        {
            return prefix + to_string(id);
        }

        public static string BatchNorm(UInt id, string inputs, string group = "", string prefix = "B")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=BatchNorm" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string LayerNorm(UInt id, string inputs, string group = "", string prefix = "LN")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=LayerNorm" + nwl +
                "Inputs=" + inputs + nwl +
                "Eps=1e-6" + nwl + nwl;
        }

        // public static string BatchNormActivation(UInt id, string inputs, string activation = "Relu", string group = "", string prefix = "B")
        // {
        //    if (activation == "Relu")
        //        return 
        //            "[" + group + prefix + to_string(id) + "]" + nwl +
        //            "Type=BatchNormRelu" + nwl +
        //            "Inputs=" + inputs + nwl + nwl;
        //    else
        //        return 
        //            "[" + group + prefix + to_string(id) + "]" + nwl +
        //            "Type=BatchNormActivation" + nwl +
        //            "Inputs=" + inputs + nwl + 
        //            "Activation=" + activation + nwl + nwl;
        // }

        public static string BatchNormActivation(UInt id, string inputs, Activations activation = Activations.Relu, string group = "", string prefix = "B")
        {
            if (activation != Activations.FRelu)
            {
                if (activation == Activations.Relu)
                {
                    return "[" + group + prefix + to_string(id) + "]" + nwl +
                        "Type=BatchNormRelu" + nwl +
                        "Inputs=" + inputs + nwl + nwl;
                }
                else
                {
                    return "[" + group + prefix + to_string(id) + "]" + nwl +
                        "Type=BatchNormActivation" + nwl +
                        "Inputs=" + inputs + nwl +
                        "Activation=" + activation.ToString() + nwl + nwl;

                    //return "[" + group + "BN" + to_string(id) + "]" + nwl +
                    //    "Type=BatchNorm" + nwl +
                    //	"Inputs=" + inputs + nwl + nwl +
                    //	"[" + group + prefix + to_string(id) + "]" + nwl +
                    //	"Type=Activation" + nwl +
                    //	"Inputs=" + group + "BN" + to_string(id) + nwl +
                    //	"Activation=" + activation.ToString() + nwl + nwl;
                }
            }
            else
            {
                return "[" + group + "B" + to_string(id) + "B1]" + nwl +
                    "Type=BatchNorm" + nwl +
                    "Inputs=" + inputs + nwl + nwl +

                    "[" + group + "DC" + to_string(id) + "DC]" + nwl +
                    "Type=DepthwiseConvolution" + nwl +
                    "Inputs=" + group + "B" + to_string(id) + "B1" + nwl +
                    "Kernel=3,3" + nwl +
                    "Pad=1,1" + nwl + nwl +

                    "[" + group + "B" + to_string(id) + "B2]" + nwl +
                    "Type=BatchNorm" + nwl +
                    "Inputs=" + group + "DC" + to_string(id) + "DC" + nwl + nwl +

                    "[" + group + prefix + to_string(id) + "]" + nwl +
                    "Type=Max" + nwl +
                    "Inputs=" + group + "B" + to_string(id) + "B2," + group + "B" + to_string(id) + "B1" + nwl + nwl;
            }
        }

        public static string BatchNormActivationDropout(UInt id, string inputs, Activations activation = Activations.Relu, Float dropout = 0.0f, string group = "", string prefix = "B")
        {
            if (activation != Activations.FRelu)
            {
                return
                    "[" + group + prefix + to_string(id) + "]" + nwl +
                    "Type=BatchNormActivationDropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    "Activation=" + activation.ToString() + nwl +
                    (dropout > 0f ? "Dropout=" + to_string(dropout) + nwl + nwl : nwl);
            }
            else
            {
                return
                    "[" + group + prefix + to_string(id) + "]" + nwl +
                    "Type=BatchNormActivationDropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    "Activation=HardSwish" + nwl +
                    (dropout > 0f ? "Dropout=" + to_string(dropout) + nwl + nwl : nwl);
            }
        }

        public static string Resampling(UInt id, string inputs, string group = "", string prefix = "R")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Resampling" + nwl +
                "Inputs=" + inputs + nwl +
                "Factor=0.5,0.5" + nwl +
                "Algorithm=Linear" + nwl + nwl;
        }

        public static string ReductionAvg(UInt id, string inputs, string group = "", string prefix = "RAVG")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Reduction" + nwl +
                "Inputs=" + inputs + nwl +
                "Operation=Avg" + nwl + nwl;
        }

        public static string ReductionMax(UInt id, string inputs, string group = "", string prefix = "RMAX")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Reduction" + nwl +
                "Inputs=" + inputs + nwl +
                "Operation=Max" + nwl + nwl;
        }

        public static string Convolution(UInt id, string inputs, UInt channels, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, string group = "", string prefix = "C", string weightsFiller = "")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Convolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + to_string(channels) + nwl +
                "Kernel=" + to_string(kernelX) + "," + to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + to_string(strideX) + "," + to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + to_string(padX) + "," + to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        public static string DepthwiseConvolution(UInt id, string inputs, UInt multiplier = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, string group = "", string prefix = "DC", string weightsFiller = "")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=DepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                (multiplier > 1 ? "Multiplier=" + to_string(multiplier) + nwl : "") +
                "Kernel=" + to_string(kernelX) + "," + to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + to_string(strideX) + "," + to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + to_string(padX) + "," + to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        public static string PartialDepthwiseConvolution(UInt id, string inputs, UInt part = 1, UInt groups = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, string group = "", string prefix = "DC", string weightsFiller = "")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=PartialDepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Group=" + to_string(part) + nwl +
                "Groups=" + to_string(groups) + nwl +
                "Kernel=" + to_string(kernelX) + "," + to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + to_string(strideX) + "," + to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + to_string(padX) + "," + to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        public static string DepthwiseMixedConvolution(UInt g, UInt id, string inputs, UInt strideX = 1, UInt strideY = 1, bool biases = false, bool useChannelSplit = true, string group = "", string prefix = "DC")
        {
            switch (g)
            {
                case 0:
                    return DepthwiseConvolution(id, inputs, 1, 3, 3, strideX, strideY, 1, 1, biases, group, prefix);

                case 1:
                    return useChannelSplit ? ChannelSplit(id, inputs, 2, 1, "Q1") + ChannelSplit(id, inputs, 2, 2, "Q2") +
                        DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix) :
                        PartialDepthwiseConvolution(id, inputs, 1, 2, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 2, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix);
                /*
                case 2:
                    return useChannelSplit ? ChannelSplit(id, inputs, 3, 1, "Q1") + ChannelSplit(id, inputs, 3, 2, "Q2") + ChannelSplit(id, inputs, 3, 3, "Q3") +
                        DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") + DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, biases, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix) :
                        PartialDepthwiseConvolution(id, inputs, 1, 3, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 3, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        PartialDepthwiseConvolution(id, inputs, 3, 3, 7, 7, strideX, strideY, 3, 3, biases, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix);
                */

                default:
                    return useChannelSplit ? ChannelSplit(id, inputs, 4, 1, "Q1") + ChannelSplit(id, inputs, 4, 2, "Q2") + ChannelSplit(id, inputs, 4, 3, "Q3") + ChannelSplit(id, inputs, 4, 4, "Q4") +
                        DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, biases, "C") + DepthwiseConvolution(id, In("Q4CS", id), 1, 9, 9, strideX, strideY, 4, 4, biases, "D") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix) :
                        PartialDepthwiseConvolution(id, inputs, 1, 4, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 4, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        PartialDepthwiseConvolution(id, inputs, 3, 4, 7, 7, strideX, strideY, 3, 3, biases, "C") + PartialDepthwiseConvolution(id, inputs, 4, 4, 9, 9, strideX, strideY, 4, 4, biases, "D") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix);
            }
        }

        public static string ChannelSplitRatioLeft(UInt id, string inputs, Float ratio = 0.375f, string group = "", string prefix = "CSRL")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=ChannelSplitRatioLeft" + nwl +
                "Inputs=" + inputs + nwl +
                "Ratio=" + to_string(ratio) + nwl + nwl;
        }

        public static string ChannelSplitRatioRight(UInt id, string inputs, Float ratio = 0.375f, string group = "", string prefix = "CSRR")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=ChannelSplitRatioRight" + nwl +
                "Inputs=" + inputs + nwl +
                "Ratio=" + to_string(ratio) + nwl + nwl;
        }

        public static string ChannelSplit(UInt id, string inputs, UInt groups, UInt part, string group = "", string prefix = "CS")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=ChannelSplit" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + to_string(groups) + nwl +
                "Group=" + to_string(part) + nwl + nwl;
        }

        public static string Shuffle(UInt id, string inputs, UInt groups = 2, string group = "", string prefix = "SH")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Shuffle" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + to_string(groups) + nwl + nwl;
        }

        public static string Concat(UInt id, string inputs, string group = "", string prefix = "CC")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Concat" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string AvgPooling(UInt id, string input, string kernel = "3,3", string stride = "2,2", string pad = "1,1", string group = "", string prefix = "P")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=AvgPooling" + nwl +
                "Inputs=" + input + nwl +
                "Kernel=" + kernel + nwl +
                "Stride=" + stride + nwl +
                "Pad=" + pad + nwl + nwl;
        }

        public static string GlobalAvgPooling(string input, string group = "", string prefix = "GAP")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=GlobalAvgPooling" + nwl +
                "Inputs=" + input + nwl + nwl;
        }

        public static string GlobalMaxPooling(string input, string group = "", string prefix = "GMP")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=GlobalMaxPooling" + nwl +
                "Inputs=" + input + nwl + nwl;
        }

        public static string Dense(UInt id, string inputs, UInt channels, bool biases = false, string group = "", string prefix = "DS", string weightsFiller = "")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Dense" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + to_string(channels) + nwl +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        public static string Add(UInt id, string inputs, string group = "", string prefix = "A")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Add" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string Multiply(string inputs, string group = "", string prefix = "CM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Multiply" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string Dropout(UInt id, string inputs, string group = "", string prefix = "D")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Dropout" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string Softmax(UInt id, string inputs, string group = "", string prefix = "SM")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Softmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string Softmax(string inputs, string group = "", string prefix = "SM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Softmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string LogSoftmax(UInt id, string inputs, string group = "", string prefix = "LSM")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=LogSoftmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string LogSoftmax(string inputs, string group = "", string prefix = "LSM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=LogSoftmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        public static string Activation(UInt id, string inputs, string activation = "Relu", string group = "", string prefix = "ACT")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Activation" + nwl +
                "Inputs=" + inputs + nwl +
                "Activation=" + activation + nwl + nwl;
        }

        public static string Activation(UInt id, string inputs, Activations activation = Activations.Relu, string group = "", string prefix = "ACT")
        {
            return "[" + group + prefix + to_string(id) + "]" + nwl +
                "Type=Activation" + nwl +
                "Inputs=" + inputs + nwl +
                "Activation=" + activation.ToString() + nwl + nwl;
        }

        public static string Cost(string inputs, Datasets dataset, UInt channels, string cost = "CategoricalCrossEntropy", Float eps = 0.0f, string group = "", string prefix = "Cost")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Cost" + nwl +
                "Inputs=" + inputs + nwl +
                "Cost=" + cost + nwl +
                "LabelIndex=" + ((dataset == Datasets.cifar100 && channels == 100) ? "1" : "0") + nwl +
                "Channels=" + to_string(channels) + nwl +
                "Eps=" + to_string(eps);
        }

        public static List<string> FusedMBConv(UInt A, UInt C, string inputs, UInt inputChannels, UInt outputChannels, UInt stride = 1, UInt expandRatio = 4, bool se = false, Activations activation = Activations.HardSwish)
        {
            var blocks = new List<string>();
            var hiddenDim = DIV8(inputChannels * expandRatio);
            var identity = stride == 1 && inputChannels == outputChannels;

            if (se)
            {
                var group = In("SE", C);

                blocks.Add(
                    Convolution(C, inputs, hiddenDim, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C, In("C", C), activation) : BatchNormActivation(C, In("C", C), activation)) +

                    GlobalAvgPooling(In("B", C), group) +
                    Convolution(1, group + "GAP", DIV8(hiddenDim / expandRatio), 1, 1, 1, 1, 0, 0, false, group) +
                    BatchNormActivation(1, group + "C1", (activation == Activations.FRelu ? Activations.HardSwish : activation), group) +
                    Convolution(2, group + "B1", hiddenDim, 1, 1, 1, 1, 0, 0, false, group) +
                    BatchNormActivation(2, group + "C2", Activations.HardSigmoid, group) +
                    Multiply(In("B", C) + "," + group + "B2", group) +

                    Convolution(C + 1, group + "CM", DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C + 1)));
            }
            else
            {
                blocks.Add(
                    Convolution(C, inputs, hiddenDim, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C, In("C", C), activation) : BatchNormActivation(C, In("C", C), activation)) +
                    Convolution(C + 1, In("B", C), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C + 1)));
            }

            if (identity)
            {
                blocks.Add(
                    Add(A, In("B", C + 1) + "," + inputs));
            }

            return blocks;
        }

        public static List<string> MBConv(UInt A, UInt C, string inputs, UInt inputChannels, UInt outputChannels, UInt stride = 1, UInt expandRatio = 4, bool se = false, Activations activation = Activations.HardSwish)
        {
            var blocks = new List<string>();
            var hiddenDim = DIV8(inputChannels * expandRatio);
            var identity = stride == 1 && inputChannels == outputChannels;

            if (se)
            {
                var group = In("SE", C + 1);

                blocks.Add(
                    Convolution(C, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C), 1, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C + 1, In("DC", C + 1), activation) : BatchNormActivation(C + 1, In("DC", C + 1), activation)) +

                    GlobalAvgPooling(In("B", C + 1), group) +
                    Convolution(1, group + "GAP", DIV8(hiddenDim / expandRatio), 1, 1, 1, 1, 0, 0, false, group) +
                    BatchNormActivation(1, group + "C1", (activation == Activations.FRelu ? Activations.HardSwish : activation), group) +
                    Convolution(2, group + "B1", hiddenDim, 1, 1, 1, 1, 0, 0, false, group) +
                    BatchNormActivation(2, group + "C2", Activations.HardSigmoid, group) +
                    Multiply(In("B", C + 1) + "," + group + "B2", group) +

                    Convolution(C + 2, group + "CM", DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 2, In("C", C + 2)));
            }
            else
            {
                blocks.Add(
                    Convolution(C, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C), 1, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C + 1, In("DC", C + 1), activation) : BatchNormActivation(C + 1, In("DC", C + 1), activation)) +
                    Convolution(C + 2, In("B", C + 1), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 2, In("C", C + 2)));
            }

            if (identity)
            {
                blocks.Add(
                    Add(A, In("B", C + 2) + "," + inputs));
            }

            return blocks;
        }


        public static string InvertedResidual(UInt A, UInt C, UInt channels, UInt kernel = 3, UInt pad = 1, bool subsample = false, UInt shuffle = 2, bool se = false, Activations activation = Activations.HardSwish)
        {
            if (subsample)
            {
                return
                    Convolution(C, In("CC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 1, In("DC", C + 1)) +
                    BatchNorm(C + 2, In("R", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    DepthwiseConvolution(C + 3, In("CC", A), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 3, In("DC", C + 3)) +
                    BatchNorm(C + 4, In("R", C + 3)) +
                    Convolution(C + 4, In("B", C + 4), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 5, In("C", C + 4), activation) +
                    Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3));
            }
            else
            {
                var groupCH = In("CHATT", C + 3); // Channel Attention
                var groupSP = In("SPATT", C + 3); // Spatial Attention
                var strSE = se ?
                    GlobalAvgPooling(In("B", C + 3), groupCH) +
                    Convolution(1, groupCH + "GAP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(1, groupCH + In("C", 1), activation, groupCH) +
                    GlobalMaxPooling(In("B", C + 3), groupCH) +
                    Convolution(2, groupCH + "GMP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(2, groupCH + In("C", 2), activation, groupCH) +
                    Add(1, In(groupCH + "B", 1) + "," + In(groupCH + "B", 2), groupCH) +
                    Convolution(3, groupCH + "A1", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(3, groupCH + In("C", 3), Activations.HardSigmoid, groupCH) +
                    Multiply(In("B", C + 3) + "," + In(groupCH + "B", 3), groupCH) +
                    ReductionAvg(1, groupCH + "CM", groupSP) +
                    ReductionMax(1, groupCH + "CM", groupSP) +
                    Concat(1, In(groupSP + "RAVG", 1) + "," + In(groupSP + "RMAX", 1), groupSP) +
                    Convolution(1, groupSP + In("CC", 1), 1, 7, 7, 1, 1, 3, 3, false, groupSP) +
                    BatchNormActivation(1, groupSP + In("C", 1), Activations.HardSigmoid, groupSP) +
                    Multiply(groupCH + "CM," + groupSP + In("B", 1), groupSP) +
                    Concat(A + 1, In("LCS", A) + "," + groupSP + "CM") :
                    Concat(A + 1, In("LCS", A) + "," + In("B", C + 3));

                return
                    Shuffle(A, In("CC", A), shuffle) +
                    ChannelSplit(A, In("SH", A), 2, 1, "L") + ChannelSplit(A, In("SH", A), 2, 2, "R") +
                    Convolution(C, In("RCS", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    BatchNorm(C + 2, In("DC", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    strSE;
            }
        }

        public static string AugmentedInvertedResidual(UInt A, UInt C, UInt channels, UInt kernel = 3, UInt pad = 1, bool subsample = false, UInt shuffle = 2, bool se = false, Activations activation = Activations.HardSwish)
        {
            if (subsample)
            {
                return
                    Convolution(C, In("CC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 1, In("DC", C + 1)) +
                    BatchNorm(C + 2, In("R", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    DepthwiseConvolution(C + 3, In("CC", A), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 3, In("DC", C + 3)) +
                    BatchNorm(C + 4, In("R", C + 3)) +
                    Convolution(C + 4, In("B", C + 4), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 5, In("C", C + 4), activation) +
                    Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3));
            }
            else
            {
                var groupCH = In("CHATT", C + 3); // Channel Attention
                var groupSP = In("SPATT", C + 3); // Spatial Attention
                var strSE = se ?
                    GlobalAvgPooling(In("B", C + 3), groupCH) +
                    Convolution(1, groupCH + "GAP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(1, groupCH + In("C", 1), activation, groupCH) +
                    GlobalMaxPooling(In("B", C + 3), groupCH) +
                    Convolution(2, groupCH + "GMP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(2, groupCH + In("C", 2), activation, groupCH) +
                    Add(1, In(groupCH + "B", 1) + "," + In(groupCH + "B", 2), groupCH) +
                    Convolution(3, groupCH + "A1", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(3, groupCH + In("C", 3), Activations.HardSigmoid, groupCH) +
                    Multiply(In("B", C + 3) + "," + In(groupCH + "B", 3), groupCH) +
                    ReductionAvg(1, groupCH + "CM", groupSP) +
                    ReductionMax(1, groupCH + "CM", groupSP) +
                    Concat(1, In(groupSP + "RAVG", 1) + "," + In(groupSP + "RMAX", 1), groupSP) +
                    Convolution(1, groupSP + In("CC", 1), 1, 7, 7, 1, 1, 3, 3, false, groupSP) +
                    BatchNormActivation(1, groupSP + In("C", 1), Activations.HardSigmoid, groupSP) +
                    Multiply(groupCH + "CM," + groupSP + In("B", 1), groupSP) +
                    Concat(A + 1, In("LCC", A) + "," + groupSP + "CM") :
                    Concat(A + 1, In("LCC", A) + "," + In("B", C + 3));

                return
                    Shuffle(A, In("CC", A), shuffle) +
                    ChannelSplitRatioLeft(A, In("SH", A), 0.375f) + ChannelSplitRatioRight(A, In("SH", A), 0.375f) +
                    Convolution(C, In("CSRR", A), DIV8((UInt)((2 * channels) * 0.375f)), 1, 1, 1, 1, 0, 0) +
                    // BatchNorm(C + 1, In("C", C)) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    BatchNorm(C + 2, In("DC", C + 1)) +
                    ChannelSplit(A, In("B", C + 2), 2, 1, "L1") + ChannelSplit(A, In("B", C + 2), 2, 2, "R1") +
                    ChannelSplit(A, In("CSRL", A), 2, 1, "L2") + ChannelSplit(A, In("CSRL", A), 2, 2, "R2") +
                    Concat(A, In("L1CS", A) + "," + In("L2CS", A), "L") +
                    Concat(A, In("R1CS", A) + "," + In("R2CS", A), "R") +
                    Convolution(C + 2, In("RCC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    strSE;
            }
        }

        internal static string Generate(ScriptParameters p)
        {
            var net =
                "[" + p.ModelName + "]" + nwl +
                "Dataset=" + to_string(p.Dataset) + nwl +
                "Dim=" + to_string(p.C) + "," + to_string(p.H) + "," + to_string(p.W) + nwl +
                ((p.PadH > 0 || p.PadW > 0) ? (!p.MirrorPad ? "ZeroPad=" + to_string(p.PadH) + "," + to_string(p.PadW) + nwl : "MirrorPad=" + to_string(p.PadH) + "," + to_string(p.PadW) + nwl) : "") +
                ((p.PadH > 0 || p.PadW > 0) ? "RandomCrop=Yes" + nwl : "") +
                "WeightsFiller=" + to_string(p.WeightsFiller) + (p.WeightsFillerModeVisible ? "(" + p.WeightsFillerMode.ToString() + "," + to_string(p.WeightsGain) + ")" : "") + (p.WeightsGainVisible && !p.WeightsFillerModeVisible ? "(" + to_string(p.WeightsGain) + ")" : "") + (p.WeightsScaleVisible ? "(" + to_string(p.WeightsScale) + ")" : "") + nwl +
                (p.WeightsLRM != 1 ? "WeightsLRM=" + to_string(p.WeightsLRM) + nwl : "") +
                (p.WeightsWDM != 1 ? "WeightsWDM=" + to_string(p.WeightsWDM) + nwl : "") +
                (p.HasBias ? "BiasesFiller=" + to_string(p.BiasesFiller) + (p.BiasesFillerModeVisible ? "(" + p.BiasesFillerMode.ToString() + "," + to_string(p.BiasesGain) + ")" : "") + (p.BiasesGainVisible && !p.BiasesFillerModeVisible ? "(" + to_string(p.BiasesGain) + ")" : "") + (p.BiasesScaleVisible ? "(" + to_string(p.BiasesScale) + ")" : "") + nwl +
                (p.BiasesLRM != 1 ? "BiasesLRM=" + to_string(p.BiasesLRM) + nwl : "") +
                (p.BiasesWDM != 1 ? "BiasesWDM=" + to_string(p.BiasesWDM) + nwl : "") : "Biases=No" + nwl) +
                (p.DropoutVisible ? "Dropout=" + to_string(p.Dropout) + nwl : "") +
                (p.DepthDropVisible ? "DepthDrop=" + to_string(p.DepthDrop) + nwl : "") +
                (p.DepthDropVisible ? "FixedDepthDrop=" + to_string(p.FixedDepthDrop) + nwl : "") +
                "Scaling=" + to_string(p.BatchNormScaling) + nwl +
                "Momentum=" + to_string(p.BatchNormMomentum) + nwl +
                "Eps=" + to_string(p.BatchNormEps) + nwl + nwl;

            var blocks = new List<string>();

            switch (p.Script)
            {
                case Scripts.augshufflenet:
                    {
                        var channels = DIV8(p.Width * 16);

                        net +=
                            Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                            BatchNormActivation(1, "C1", p.Activation) +
                            Convolution(2, "B1", channels, 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(2, "C2", p.Activation) +
                            DepthwiseConvolution(3, "B2", 1, 3, 3, 1, 1, 1, 1) +
                            BatchNorm(3, "DC3") +
                            Convolution(4, "B3", channels, 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(4, "C4", p.Activation) +
                            Convolution(5, "B1", channels, 1, 1, 1, 1, 0, 0) +
                            Concat(1, "C5,B4");

                        var C = 6ul;
                        var A = 1ul;
                        var subsample = false;
                        foreach (var rec in p.ShuffleNet)
                        {
                            if (subsample)
                            {
                                channels *= 2;
                                net += AugmentedInvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, true, rec.Shuffle, rec.SE, p.Activation);
                                C += 5;
                            }
                            for (var n = 0ul; n < rec.Iterations; n++)
                            {
                                net += AugmentedInvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, false, rec.Shuffle, rec.SE, p.Activation);
                                C += 3;
                            }
                            subsample = true;
                        }

                        net +=
                            Convolution(C, In("CC", A), p.Classes, 1, 1, 1, 1, 0, 0) +
                            BatchNorm(C + 1, In("C", C)) +
                            GlobalAvgPooling(In("B", C + 1)) +
                            LogSoftmax("GAP") +
                            Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                    }
                    break;

                case Scripts.densenet:
                    {
                        var channels = DIV8(p.GrowthRate);

                        net += Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1);

                        if (p.Bottleneck)
                        {
                            blocks.Add(
                                BatchNormActivation(1, "C1", p.Activation) +
                                Convolution(2, "B1", DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                BatchNormActivation(2, "C2", p.Activation) +
                                Convolution(3, "B2", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(3, "C3") + Concat(1, "C1,D3") : Concat(1, "C1,C3")));
                        }
                        else
                        {
                            blocks.Add(
                                BatchNormActivation(1, "C1", p.Activation) +
                                Convolution(2, "B1", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(2, "C2") + Concat(1, "C1,D2") : Concat(1, "C1,C2")));
                        }

                        var CC = 1ul;
                        var C = p.Bottleneck ? 4ul : 3ul;

                        channels += DIV8(p.GrowthRate);

                        for (var g = 1ul; g <= p.Groups; g++)
                        {
                            for (var i = 1ul; i < p.Iterations; i++)
                            {
                                if (p.Bottleneck)
                                {
                                    blocks.Add(
                                        BatchNormActivation(C, In("CC", CC), p.Activation) +
                                        Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                        BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                        Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                        (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C + 1)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C + 1))));

                                    C += 2;
                                }
                                else
                                {
                                    blocks.Add(
                                        BatchNormActivation(C, In("CC", CC), p.Activation) +
                                        Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                        (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C))));

                                    C++;
                                }

                                CC++;
                                channels += DIV8(p.GrowthRate);
                            }

                            if (g < p.Groups)
                            {
                                channels = DIV8((UInt)System.Math.Floor(2.0 * channels * p.Compression));

                                if (p.Dropout > 0)
                                    blocks.Add(
                                        Convolution(C, In("CC", CC), channels, 1, 1, 1, 1, 0, 0) +
                                        Dropout(C, In("C", C)) +
                                        AvgPooling(g, In("D", C), "2,2", "2,2", "0,0"));
                                else
                                    blocks.Add(
                                        Convolution(C, "CC" + to_string(CC), channels, 1, 1, 1, 1, 0, 0) +
                                        AvgPooling(g, In("C", C), "2,2", "2,2", "0,0"));
                                C++;
                                CC++;

                                if (p.Bottleneck)
                                {
                                    blocks.Add(
                                        BatchNormActivation(C, In("P", g), p.Activation) +
                                        Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                        BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                        Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                        (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC, In("B", C) + "," + In("D", C + 1)) : Concat(CC, In("B", C) + "," + In("C", C + 1))));

                                    C += 2;
                                }
                                else
                                {
                                    blocks.Add(
                                        BatchNormActivation(C, In("P", g), p.Activation) +
                                        Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                        (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC, In("B", C) + "," + In("D", C)) : Concat(CC, In("B", C) + "," + In("C", C))));

                                    C++;
                                }

                                channels += DIV8(p.GrowthRate);
                            }
                        }

                        foreach (var block in blocks)
                            net += block;

                        net +=
                            Convolution(C, In("CC", CC), p.Classes, 1, 1, 1, 1, 0, 0) +
                            BatchNorm(C + 1, In("C", C)) +
                            GlobalAvgPooling(In("B", C + 1)) +
                            LogSoftmax("GAP") +
                            Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                    }
                    break;

                case Scripts.efficientnetv2:
                    {
                        const Float width = 1.0f;
                        var inputChannels = DIV8((UInt)((Float)p.EfficientNet[0].Channels * width));
                        var A = 1ul;
                        var C = 1ul;

                        net +=
                            Convolution(C, "Input", inputChannels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                            BatchNormActivation(C, In("C", C), p.Activation);

                        var stage = 0ul;
                        var input = In("B", C++);
                        foreach (var rec in p.EfficientNet)
                        {
                            var beginStage = stage < 3ul;
                            var outputChannels = DIV8((UInt)((Float)rec.Channels * width));
                            for (var n = 0ul; n < rec.Iterations; n++)
                            {
                                var stride = n == 0ul ? rec.Stride : 1ul;
                                var identity = stride == 1ul && inputChannels == outputChannels;

                                var subblocks = beginStage ?
                                    FusedMBConv(A, C, input, inputChannels, outputChannels, stride, rec.ExpandRatio, rec.SE, p.Activation) :
                                    MBConv(A, C, input, inputChannels, outputChannels, stride, rec.ExpandRatio, rec.SE, p.Activation);

                                foreach (var blk in subblocks)
                                    net += blk;

                                inputChannels = outputChannels;
                                C += beginStage ? 1ul : 2ul;

                                if (identity)
                                {
                                    input = In("A", A++);
                                    C++;
                                }
                                else
                                    input = In("B", C++);
                            }
                            stage++;
                        }

                        net +=
                            Convolution(C, In("A", A - 1), p.Classes, 1, 1, 1, 1, 0, 0) +
                            BatchNormActivationDropout(C, In("C", C), p.Activation) +
                            GlobalAvgPooling(In("B", C)) +
                            LogSoftmax("GAP") +
                            Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                    }
                    break;

                case Scripts.mobilenetv3:
                    {
                        var se = p.SqueezeExcitation;
                        var channelsplit = true;
                        var W = p.Width * 16;

                        net +=
                            Convolution(1, "Input", DIV8(W), 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                            BatchNormActivation(1, "C1", p.Activation);

                        blocks.Add(
                            Convolution(2, "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(2, "C2", p.Activation) +
                            DepthwiseMixedConvolution(0, 3, "B2", 1, 1, p.HasBias, channelsplit) +
                            BatchNormActivation(3, "DC3", p.Activation) +
                            Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNorm(4, "C4"));

                        var A = 1ul;
                        var C = 5ul;
                        for (var g = 1ul; g <= p.Groups; g++)
                        {
                            var mix = g - 1ul;

                            if (g > 1)
                            {
                                W *= 2;

                                var group = In("SE", C + 1);
                                var strSE =
                                    se ? GlobalAvgPooling(In("B", C + 1), group) +
                                    Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, false, group) +
                                    BatchNormActivation(1, group + "C1", (p.Activation == Activations.FRelu ? Activations.HardSwish : p.Activation), group) +
                                    Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, false, group) +
                                    BatchNormActivation(2, group + "C2", Activations.HardSigmoid, group) +
                                    Multiply(In("B", C + 1) + "," + group + "B2", group) +
                                    Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                                    Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                                blocks.Add(
                                    Convolution(C, In("A", A), DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                                    BatchNormActivation(C, In("C", C), p.Activation) +
                                    DepthwiseMixedConvolution(1ul, C + 1, In("B", C), 2, 2, p.HasBias, channelsplit) +
                                    BatchNormActivation(C + 1, In("DC", C + 1), p.Activation) +
                                    strSE +
                                    BatchNorm(C + 2, In("C", C + 2)));

                                C += 3;
                            }

                            for (var i = 1ul; i < p.Iterations; i++)
                            {
                                var strOutputLayer = (i == 1 && g > 1) ? In("B", C - 1) : (i == 1 && g == 1) ? In("B", 4) : In("A", A);

                                var group = In("SE", C + 1);

                                var strSE =
                                    se ? GlobalAvgPooling(In("B", C + 1), group) +
                                    Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, false, group) +
                                    BatchNormActivation(1, group + "C1", (p.Activation == Activations.FRelu ? Activations.HardSwish : p.Activation), group) +
                                    Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, false, group) +
                                    BatchNormActivation(2, group + "C2", Activations.HardSigmoid, group) +
                                    Multiply(In("B", C + 1) + "," + group + "B2", group) +
                                    Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                                    Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                                blocks.Add(
                                    Convolution(C, strOutputLayer, DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                                    BatchNormActivation(C, In("C", C), p.Activation) +
                                    DepthwiseMixedConvolution(1ul, C + 1, In("B", C), 1, 1, p.HasBias, channelsplit) +
                                    BatchNormActivation(C + 1, In("DC", C + 1), p.Activation) +
                                    strSE +
                                    BatchNorm(C + 2, In("C", C + 2)) +
                                    Add(A + 1, In("B", C + 2) + "," + strOutputLayer));

                                A++;
                                C += 3;
                            }
                        }

                        foreach (var block in blocks)
                            net += block;

                        net +=
                            BatchNormActivation(C, In("A", A), p.Activation) +
                            Convolution(C + 1, In("B", C), p.Classes, 1, 1, 1, 1, 0, 0) +
                            BatchNorm(C + 1, In("C", C + 1)) +
                            GlobalAvgPooling(In("B", C + 1)) +
                            LogSoftmax("GAP") +
                            Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                    }
                    break;

                case Scripts.resnet:
                {
                    var bn = p.Bottleneck ? 1ul : 0ul;
                    const Float K = 2.0f;
                    var W = p.Width * 16;
                    var A = 1ul;
                    var C = 5ul;

                    net += Convolution(1, "Input", DIV8(W), 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1);

                    if (p.Bottleneck)
                    {
                        blocks.Add(
                            BatchNormActivation(1, "C1", p.Activation) +
                            Convolution(2, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(2, "C2", p.Activation) +
                            Convolution(3, "B2", DIV8((UInt)(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                            (p.Dropout > 0 ? BatchNormActivationDropout(3, "C3", p.Activation) : BatchNormActivation(3, "C3", p.Activation)) +
                            Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                            Convolution(5, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                            Add(1, "C4,C5"));

                        C = 6;
                    }
                    else
                    {
                        blocks.Add(
                            BatchNormActivation(1, "C1", p.Activation) +
                            Convolution(2, "B1", DIV8(W), 3, 3, 1, 1, 1, 1) +
                            (p.Dropout > 0 ? BatchNormActivationDropout(2, "C2", p.Activation) : BatchNormActivation(2, "C2", p.Activation)) +
                            Convolution(3, "B2", DIV8(W), 3, 3, 1, 1, 1, 1) +
                            Convolution(4, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                            Add(1, "C3,C4"));
                    }

                    for (var g = 0ul; g < p.Groups; g++)
                        {
                        if (g > 0)
                            {
                            W *= 2;

                            var strChannelZeroPad = p.ChannelZeroPad ?
                                AvgPooling(g, In("A", A)) +
                                "[CZP" + to_string(g) + "]" + nwl + "Type=ChannelZeroPad" + nwl + "Inputs=" + In("P", g) + nwl + "Channels=" + to_string(W) + nwl + nwl +
                                Add(A + 1, In("C", C + 1 + bn) + "," + In("CZP", g)) :
                                AvgPooling(g, In("B", C)) +
                                Convolution(C + 2 + bn, In("P", g), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                Add(A + 1, In("C", C + 1 + bn) + "," + In("C", C + 2 + bn));

                            if (p.Bottleneck)
                            {
                                blocks.Add(
                                    BatchNormActivation(C, In("A", A), p.Activation) +
                                    Convolution(C, In("B", C), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                    BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                    Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 2, 2, 1, 1) +
                                    (p.Dropout > 0 ? BatchNormActivationDropout(C + 2, In("C", C + 1), p.Activation) : BatchNormActivation(C + 2, In("C", C + 1), p.Activation)) +
                                    Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                    strChannelZeroPad);
                            }
                            else
                            {
                                blocks.Add(
                                    BatchNormActivation(C, In("A", A), p.Activation) +
                                    Convolution(C, In("B", C), DIV8(W), 3, 3, 2, 2, 1, 1) +
                                    (p.Dropout > 0 ? BatchNormActivationDropout(C + 1, In("C", C), p.Activation) : BatchNormActivation(C + 1, In("C", C), p.Activation)) +
                                    Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                    strChannelZeroPad);
                            }

                            A++;
                            C += p.ChannelZeroPad ? 2 + bn : 3 + bn;
                        }

                        for (var i = 1ul; i < p.Iterations; i++)
                            {
                            if (p.Bottleneck)
                            {
                                blocks.Add(
                                    BatchNormActivation(C, In("A", A), p.Activation) +
                                    Convolution(C, In("B", C), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                    BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                    Convolution(C + 1, In("B", C + 1), DIV8((UInt)(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                                    (p.Dropout > 0 ? BatchNormActivationDropout(C + 2, In("C", C + 1), p.Activation) : BatchNormActivation(C + 2, In("C", C + 1), p.Activation)) +
                                    Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                    Add(A + 1, In("C", C + 2) + "," + In("A", A)));

                                C += 3;
                            }
                            else
                            {
                                blocks.Add(
                                    BatchNormActivation(C, In("A", A), p.Activation) +
                                    Convolution(C, In("B", C), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                    (p.Dropout > 0 ? BatchNormActivationDropout(C + 1, In("C", C), p.Activation) : BatchNormActivation(C + 1, In("C", C), p.Activation)) +
                                    Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                    Add(A + 1, In("C", C + 1) + "," + In("A", A)));

                                C += 2;
                            }
                            A++;
                        }
                    }

                    foreach (var block in blocks)
                        net += block;

                    net +=
                        BatchNormActivation(C, In("A", A), p.Activation) +
                        Convolution(C + 1, In("B", C), p.Classes, 1, 1, 1, 1, 0, 0) +
                        BatchNorm(C + 1, In("C", C + 1)) +
                        GlobalAvgPooling(In("B", C + 1)) +
                        LogSoftmax("GAP") +
                        Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                }
                break;

            case Scripts.shufflenetv2:
                {
                    var channels = DIV8(p.Width * 16);

                    net +=
                        Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                        BatchNormActivation(1, "C1", p.Activation) +
                        Convolution(2, "B1", channels, 1, 1, 1, 1, 0, 0) +
                        BatchNormActivation(2, "C2", p.Activation) +
                        DepthwiseConvolution(3, "B2", 1, 3, 3, 1, 1, 1, 1) +
                        BatchNorm(3, "DC3") +
                        Convolution(4, "B3", channels, 1, 1, 1, 1, 0, 0) +
                        BatchNormActivation(4, "C4", p.Activation) +
                        Convolution(5, "B1", channels, 1, 1, 1, 1, 0, 0) +
                        Concat(1, "C5,B4");

                    var C = 6ul;
                    var A = 1ul;
                    var subsample = false;
                    foreach (var rec in p.ShuffleNet)
                    {
                        if (subsample)
                        {
                            channels *= 2;
                            net += InvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, true, rec.Shuffle, rec.SE, p.Activation);
                            C += 5;
                        }
                        for (var n = 0ul; n < rec.Iterations; n++)
                            {
                            net += InvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, false, rec.Shuffle, rec.SE, p.Activation);
                            C += 3;
                        }
                        subsample = true;
                    }

                    net +=
                        Convolution(C, In("CC", A), p.Classes, 1, 1, 1, 1, 0, 0) +
                        BatchNorm(C + 1, In("C", C)) +
                        GlobalAvgPooling(In("B", C + 1)) +
                        LogSoftmax("GAP") +
                        Cost("LSM", p.Dataset, p.Classes, "CategoricalCrossEntropy", 0.125f);
                }
                break;
            }

            return net;
        }


        const string Framework = "net9.0";
#if DEBUG
        const string Mode = "Debug";
#else
        const string Mode = "Release";
#endif

        public static string StorageDirectory { get; } = Path.Combine(Environment.GetFolderPath(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Environment.SpecialFolder.MyDocuments : Environment.SpecialFolder.UserProfile), "convnet");
        public static string ScriptsDirectory { get; } = Path.Combine(StorageDirectory, "scripts");
        public static string ScriptPath { get; } = Path.Combine(ScriptsDirectory, "bin", Mode, Framework);

        static void Main()
        {
            var script = Generate(new ScriptParameters()
		    {
        	    Script = Scripts.augshufflenet,
                Activation = Activations.HardSwish,
                Dataset = Datasets.cifar10,
                MeanStdNormalization = true,
                H = 32,
        	    W = 32,
        	    PadH = 4,
        	    PadW = 4,
                MirrorPad = false,
                StrideHFirstConv = 1,
                StrideWFirstConv = 1,
                WeightsFiller = Fillers.HeNormal,
        	    WeightsFillerMode = FillerModes.In,
        	    WeightsGain = 1f,
        	    WeightsScale = 0.05f,
        	    WeightsLRM = 1f,
        	    WeightsWDM = 1f,
        	    HasBias = false,
        	    BiasesFiller = Fillers.Constant,
        	    BiasesFillerMode = FillerModes.In,
        	    BiasesGain = 1f,
        	    BiasesScale = 0f,
        	    BiasesLRM = 1f,
        	    BiasesWDM = 1f,
        	    BatchNormMomentum = 0.995f,
        	    BatchNormEps = 0.0001f,
        	    BatchNormScaling = false,
        	    Alpha = 0f,
        	    Beta = 0f,
        	    Groups = 3,
        	    Iterations = 4,
        	    Width = 16,
        	    GrowthRate = 12,
        	    Bottleneck = false,
        	    Dropout = 0f,
        	    Compression = 0f,
        	    SqueezeExcitation = true,
        	    ChannelZeroPad = false,
        	    DepthDrop = 0.0f,
        	    FixedDepthDrop = false,
        	    EfficientNet = [new(1, 24, 2, 1, false), new(4, 48, 4, 2, false), new(4, 64, 4, 2, false), new(4, 128, 6, 2, true), new(6, 160, 9, 1, true), new(6, 256, 15, 2, true)],
                ShuffleNet = [new(7, 3, 1, 2, false) , new(7, 3, 1, 2, true), new(7, 3, 1, 2, true)] 
            });

            var fileInfo = new FileInfo(Path.Combine(ScriptPath, @"script.txt"));
            
            if (fileInfo.Directory != null)
            {
                if (!fileInfo.Directory.Exists)
                    fileInfo.Directory.Create();

                var streamWriter = fileInfo.CreateText();
                streamWriter.AutoFlush = true;
                streamWriter.Write(script);
                streamWriter.Close();
                streamWriter.Dispose();
            }
        }
    }
}
