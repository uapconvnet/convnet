#ifndef _WIN32
  #include <stdlib.h>
  #define DNN_API extern "C" 
#else
#ifdef DNN_DLL
  #define DNN_API extern "C" __declspec(dllimport)
#else
  #define DNN_API extern "C"
#endif
#endif

#include "Model.h"
#include "Scripts.h"

#ifdef _WIN32
static const std::string path = std::string(getenv("USERPROFILE")) + std::string("\\Documents\\convnet\\");
#else
static const std::string path = std::string(getenv("HOME")) + std::string("/convnet/");
#endif

using namespace dnn;

DNN_API bool DNNStochasticEnabled();
DNN_API void DNNSetLocked(const bool locked);
DNN_API bool DNNSetLayerLocked(const UInt layerIndex, const bool locked);
DNN_API void DNNPersistOptimizer(const bool persist);
DNN_API void DNNDisableLocking(const bool disable);
DNN_API void DNNGetConfusionMatrix(const UInt costLayerIndex, UInt* confusionMatrix);
DNN_API void DNNGetLayerInputs(const UInt layerIndex, UInt* inputs);
DNN_API void DNNGetLayerInfo(const UInt layerIndex, dnn::LayerInfo* info);
DNN_API void DNNSetNewEpochDelegate(void* newEpoch);
DNN_API void DNNModelDispose();
DNN_API void DNNDataproviderDispose();
DNN_API bool DNNBatchNormUsed();
DNN_API void DNNResetWeights();
DNN_API void DNNResetLayerWeights(const UInt layerIndex);
DNN_API void DNNAddTrainingRate(const dnn::TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples);
DNN_API void DNNAddTrainingRateSGDR(const dnn::TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt gotoCycle, const UInt trainSamples);
DNN_API void DNNClearTrainingStrategies();
DNN_API void DNNSetUseTrainingStrategy(const bool enable);
DNN_API void DNNAddTrainingStrategy(const dnn::TrainingStrategy& strategy);
DNN_API bool DNNLoadDataset();
DNN_API void DNNTraining();
DNN_API void DNNStop();
DNN_API void DNNPause();
DNN_API void DNNResume();
DNN_API void DNNTesting();
DNN_API void DNNGetTrainingInfo(dnn::TrainingInfo* info);
DNN_API void DNNGetTestingInfo(dnn::TestingInfo* info);
DNN_API void DNNGetModelInfo(dnn::ModelInfo* info);
DNN_API void DNNSetOptimizer(const dnn::Optimizers strategy);
DNN_API void DNNResetOptimizer();
DNN_API void DNNRefreshStatistics(const UInt layerIndex, dnn::StatsInfo* info);
DNN_API bool DNNGetInputSnapShot(Float* snapshot, UInt* label);
DNN_API bool DNNCheck(char* definition, dnn::CheckMsg& checkMsg);
DNN_API int DNNLoad(const char* fileName, dnn::CheckMsg& checkMsg);
DNN_API int DNNRead(const char* definition, dnn::CheckMsg& checkMsg);
DNN_API void DNNDataprovider(const char* directory);
DNN_API int DNNLoadWeights(const char* fileName, const bool persistOptimizer);
DNN_API int DNNSaveWeights(const char* fileName, const bool persistOptimizer);
DNN_API int DNNLoadLayerWeights(const char* fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API int DNNSaveLayerWeights(const char* fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API void DNNGetLayerWeights(const UInt layerIndex, Float* weights, Float* biases);
DNN_API void DNNSetCostIndex(const UInt index);
DNN_API void DNNGetCostInfo(const UInt costIndex, dnn::CostInfo* info);
DNN_API void DNNGetImage(const UInt layer, const Byte fillColor, Byte* image);
DNN_API bool DNNSetFormat(const bool plain);
DNN_API dnn::Optimizers GetOptimizer();
DNN_API bool DNNClearLog();
//DNN_API void DNNPrintModel(const char* fileName);

std::string ToTime(UInt nanoseconds)
{
    auto seconds = nanoseconds / 1000000000ull;
    auto hours = seconds / 3600ull;
    auto minutes = (seconds - (hours * 3600ull)) / 60ull;
    seconds = (seconds - (hours * 3600ull)) - (minutes * 60ull);

    return  ((hours <  10ull ? std::string("0") : std::string("")) + std::to_string(hours) + std::string(":") + (minutes < 10ull ? std::string("0") : std::string("")) + std::to_string(minutes) + std::string(":") + (seconds < 10ull ? std::string("0") : std::string("")) + std::to_string(seconds));
}

void NewEpoch(UInt CurrentCycle, UInt CurrentEpoch, UInt TotalEpochs, UInt Optimizer, Float Beta2, Float Gamma, Float Eps, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, UInt Interpolation, Float Scaling, Float Rotation, Float MaximumRate, UInt N, UInt D, UInt H, UInt W, UInt PadD, UInt PadH, UInt PadW, Float Momentum, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, UInt ElapsedNanoSeconds)
{
    std::cout << std::string("Cycle: ") << std::to_string(CurrentCycle) << std::string("  Epoch: ") << std::to_string(CurrentEpoch) << std::string("  Train Accuracy: ") << FloatToStringFixed(TrainAccuracy, 2) << std::string("%  Test Accuracy: ") << FloatToStringFixed(TestAccuracy, 2) << std::string("%  Duration: ") + ToTime(ElapsedNanoSeconds) + std::string("                                                                           ") << std::endl;
    std::cout.flush();

    DNN_UNREF_PAR(TotalEpochs);
    DNN_UNREF_PAR(Optimizer);
    DNN_UNREF_PAR(Beta2);
    DNN_UNREF_PAR(Eps);
    DNN_UNREF_PAR(HorizontalFlip);
    DNN_UNREF_PAR(VerticalFlip);
    DNN_UNREF_PAR(InputDropout);
    DNN_UNREF_PAR(Cutout);
    DNN_UNREF_PAR(CutMix);
    DNN_UNREF_PAR(AutoAugment);
    DNN_UNREF_PAR(ColorCast);
    DNN_UNREF_PAR(ColorAngle);
    DNN_UNREF_PAR(Distortion);
    DNN_UNREF_PAR(Interpolation);
    DNN_UNREF_PAR(Scaling);
    DNN_UNREF_PAR(Rotation);
    DNN_UNREF_PAR(MaximumRate);
    DNN_UNREF_PAR(N);
    DNN_UNREF_PAR(D);
    DNN_UNREF_PAR(H);
    DNN_UNREF_PAR(W);
    DNN_UNREF_PAR(PadD);
    DNN_UNREF_PAR(PadH);
    DNN_UNREF_PAR(PadW);
    DNN_UNREF_PAR(Momentum);
    DNN_UNREF_PAR(L2Penalty);
    DNN_UNREF_PAR(Gamma);
    DNN_UNREF_PAR(Dropout);
    DNN_UNREF_PAR(AvgTrainLoss);
    DNN_UNREF_PAR(TrainErrorPercentage);
    DNN_UNREF_PAR(TrainAccuracy);
    DNN_UNREF_PAR(TrainErrors);
    DNN_UNREF_PAR(AvgTestLoss);
    DNN_UNREF_PAR(TestErrorPercentage);
    DNN_UNREF_PAR(TestErrors);
}

void GetTrainingProgress(int seconds = 5, UInt trainingSamples = 50000, UInt testingSamples = 10000)
{
    auto info = new dnn::TrainingInfo();
   
    info->State = States::Idle;
    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        DNNGetTrainingInfo(info);
    } 
    while (info->State == States::Idle);

    int barWidth = 40;
    float progress = 0.0f;
  
    while (info->State != States::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        
        DNNGetTrainingInfo(info);
       
        if (info->State == States::Testing || info->State == States::Training)
        {
            if (info->State == States::Testing)
                progress = Float(info->SampleIndex) / testingSamples;
            else
                progress = Float(info->SampleIndex) / trainingSamples;

            std::cout << std::string("[");
            int pos = int(barWidth * progress);
            for (int i = 0; i < barWidth; ++i)
            {
                if (i < pos)
                    std::cout << std::string("=");
                else
                    if (i == pos)
                        std::cout << std::string(">");
                    else
                        std::cout << std::string(" ");
            }
            std::cout << std::string("] ") << FloatToStringFixed(progress * 100.0f, 2) << std::string("%  Cycle:") << std::to_string(info->Cycle) << std::string("  Epoch:") << std::to_string(info->Epoch) << std::string("  Error:");

            if (info->State == States::Testing)
                std::cout << FloatToStringFixed(info->TestErrorPercentage, 2);
            else
                std::cout << FloatToStringFixed(info->TrainErrorPercentage, 2);

            std::cout << std::string("%  ") << FloatToStringFixed(info->SampleSpeed, 2) << std::string(" samples/s   \r");
            std::cout.flush();
        }
    }
   
    delete info;
}


#ifdef _WIN32
int __cdecl wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    auto gotoEpoch = 1ull;
    auto gotoCycle = 1ull;

    if (argc == 2)
    {
        try
        {
#ifdef _WIN32
            gotoEpoch = static_cast<UInt>(_wtoll(argv[1]));
#else
            gotoEpoch = static_cast<UInt>(atoll(argv[1]));
#endif
        }
        catch (std::exception e) 
        {
            return EXIT_FAILURE;
        }

    }

    if (argc == 3)
    {
        try
        {
#ifdef _WIN32
            gotoEpoch = static_cast<UInt>(_wtoll(argv[1]));
            gotoCycle = static_cast<UInt>(_wtoll(argv[2]));
#else
            gotoEpoch = static_cast<UInt>(atoll(argv[1]));
            gotoCycle = static_cast<UInt>(atoll(argv[2]));
#endif
        }
        catch (std::exception e)
        {
            return EXIT_FAILURE;
        }
    }

    gotoEpoch = gotoEpoch < 1ull ? 1ull : gotoEpoch;
    gotoCycle = gotoCycle < 1ull ? 1ull : gotoCycle;
   

    CheckMsg msg;

    scripts::ScriptParameters p;

    p.Script = scripts::Scripts::augshufflenet;
    p.Dataset = scripts::Datasets::cifar10;
    p.C = 3;
    p.H = 32;
    p.W = 32;
    p.PadH = 4;
    p.PadW = 4;
    p.MirrorPad = false;
    p.Groups = 3;
    p.Iterations = 4;
    p.Width = 12;
    p.Activation = scripts::Activations::HardSwish;
    p.Dropout = Float(0);
    p.Bottleneck = false;
    p.SqueezeExcitation = true;
    p.ChannelZeroPad = false;
    p.EfficientNet = { { 1, 24, 2, 1, false }, { 4, 48, 4, 2, false }, { 4, 64, 4, 2, false }, { 4, 128, 6, 2, true }, { 6, 160, 9, 1, true }, { 6, 256, 15, 2, true } };
    p.ShuffleNet = { { 7, 3, 1, 2, false }, { 7, 3, 1, 2, true }, { 7, 3, 1, 2, true } };
    p.WeightsFiller = scripts::Fillers::HeNormal;
    p.WeightsFillerMode = scripts::FillerModes::In;
    p.StrideHFirstConv = 1;
    p.StrideWFirstConv = 1;

    auto model = scripts::ScriptsCatalog::Generate(p);

    const auto persistOptimizer = true;
    const auto optimizer = Optimizers::NAG;
           
    dnn::TrainingRate rate;
    rate.Optimizer = optimizer;
    rate.Momentum = Float(0.9);
    rate.Beta2 = Float(0.999);
    rate.L2Penalty = Float(0.0005);
    rate.Dropout = Float(0.0);
    rate.Eps = Float(0.00001),
    rate.N = 128;
    rate.D = 1;
    rate.H = 32;
    rate.W = 32;
    rate.PadD = 0;
    rate.PadH = 4;
    rate.PadW = 4;
    rate.Cycles = 1;
    rate.Epochs = 200;
    rate.EpochMultiplier = 1;
    rate.MaximumRate = Float(0.05);
    rate.MinimumRate = Float(0.0001);
    rate.FinalRate = Float(0.1);
    rate.Gamma = Float(0.003);
    rate.DecayAfterEpochs = 200;
    rate.DecayFactor = Float(1.0);
    rate.HorizontalFlip = true;
    rate.VerticalFlip = false;
    rate.InputDropout = Float(0.0);
    rate.Cutout = Float(0.7);
    rate.CutMix = true;
    rate.ColorAngle = 16;
    rate.ColorCast = Float(0.7);
    rate.AutoAugment = Float(0.7);
    rate.Distortion = Float(0.7);
    rate.Interpolation = dnn::Interpolations::Cubic;
    rate.Scaling = Float(10.0);
    rate.Rotation = Float(12.0);
    
    DNNDataprovider(path.c_str());
    
    if (DNNRead(model.c_str(), msg) == 1)
    {
        if (DNNLoadDataset())
        {
            DNNResetWeights();

            //DNNPrintModel((path + std::string("Normal.txt")).c_str());

            auto info = new ModelInfo();
            DNNGetModelInfo(info);
            
            DNNSetNewEpochDelegate(reinterpret_cast<void*>(&NewEpoch));
            
            DNNSetFormat(false);
            DNNPersistOptimizer(persistOptimizer);
            DNNSetOptimizer(optimizer);
            DNNSetUseTrainingStrategy(false);
            DNNSetLocked(false);

            const auto& dir = std::filesystem::path(std::filesystem::u8path(path)) / std::string("definitions") / p.GetName();
            if (gotoEpoch == 1ull && gotoCycle == 1ull)
                DNNClearLog();
            else
                for (auto const& dir_entry : std::filesystem::directory_iterator{ dir })
                    if (dir_entry.is_directory())
                    {
                        const auto& entry = dir_entry.path().string();
                        const auto& dirname = persistOptimizer ? (std::string("(") + StringToLower(std::string(magic_enum::enum_name<scripts::Datasets>(p.Dataset))) + std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(optimizer))) + std::string(")") + std::to_string((gotoEpoch - 1)) + std::string("-") + std::to_string(gotoCycle) + std::string("-")) : (std::string("(") + StringToLower(std::string(magic_enum::enum_name<scripts::Datasets>(p.Dataset))) + std::string(")") + std::to_string((gotoEpoch - 1)) + std::string("-") + std::to_string(gotoCycle) + std::string("-"));
#ifndef NDEBUG
                        std::cerr << entry << std::endl;
                        std::cerr << dirname << std::endl;
#endif
                        if (entry.find(dirname) != std::string::npos)
                            for (auto const& subdir_entry : std::filesystem::directory_iterator{ dir_entry.path() })
                                if (subdir_entry.is_regular_file())
                                {
                                    const auto& filename = subdir_entry.path().string();
                                    const auto& compare = persistOptimizer ? (std::string("(") + StringToLower(std::string(magic_enum::enum_name<scripts::Datasets>(p.Dataset))) + std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(optimizer))) + std::string(").bin")) : (std::string("(") + StringToLower(std::string(magic_enum::enum_name<scripts::Datasets>(p.Dataset))) + std::string(").bin"));
#ifndef NDEBUG
                                    std::cerr << filename << std::endl;
                                    std::cerr << compare << std::endl;
#endif
                                    if (filename.find(compare) != std::string::npos)
                                    {
#ifndef NDEBUG
                                        std::cerr << std::string("Loading...") << std::endl;
#endif
                                        if (DNNLoadWeights(filename.c_str(), persistOptimizer) == 0)
#ifndef NDEBUG
                                            std::cerr << std::string("Loaded") << std::endl;
#else
                                            ;
#endif
                                        else
#ifndef NDEBUG
                                            std::cerr << std::string("Not loaded") << std::endl;
#else
                                            ;
#endif
                                        break;
                                    }
                                }
                    }

            std::cout << std::string("Training ") << info->Name << std::string(" on ") << std::string(magic_enum::enum_name<Datasets>(info->Dataset)) << (std::string(" with ") + std::string(magic_enum::enum_name<Optimizers>(optimizer)) + std::string(" optimizer")) << std::endl << std::endl;
            std::cout.flush();
                        
            DNNAddTrainingRateSGDR(rate, true, gotoEpoch, gotoCycle, info->TrainSamplesCount);
            DNNTraining();
            GetTrainingProgress(5, info->TrainSamplesCount, info->TestSamplesCount);
            
            delete info;
                   
            DNNStop();
            DNNModelDispose();
        }
        else
            std::cout << std::endl << std::string("Could not load dataset") << std::endl;
    }
    else
        std::cout << std::endl << std::string("Could not load model") << std::endl << msg.Message << std::endl << model << std::endl;

    DNNDataproviderDispose();

    return EXIT_SUCCESS;
}