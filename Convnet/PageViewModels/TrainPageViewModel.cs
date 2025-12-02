using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Data;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Convnet.Common;
using Convnet.Dialogs;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Legends;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using Float = System.Single;
using UInt = System.UInt64;

namespace Convnet.PageViewModels
{
    [Serializable]
    public enum PlotType
    {
        Accuracy = 0,
        Error = 1,
        Loss = 2
    };

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
    public class TrainPageViewModel : PageViewModelBase
    {
        private static readonly string nwl              = Environment.NewLine;

        private readonly string stringTesting           = "Testing" + nwl + 
                                                          " Sample:\t\t\t{0:G}" + nwl + 
                                                          " Cycle:\t\t\t {1}/{2}" + nwl + 
                                                          " Epoch:\t\t\t {3}/{4}" + nwl + 
                                                          " Batch Size:\t\t{5:G}" + nwl + 
                                                          " Loss:\t\t\t  {6:N7}" + nwl + 
                                                          " Errors:\t\t\t{7:G}" + nwl + 
                                                          " Error:\t\t\t {8:N2} %" + nwl + 
                                                          " Accuracy:\t\t  {9:N2} %";
        private readonly string stringTraining          = "Training" + nwl + 
                                                          " Sample:\t\t\t{0:G}" + nwl + 
                                                          " Cycle:\t\t\t {1}/{2}" + nwl + 
                                                          " Epoch:\t\t\t {3}/{4}" + nwl + 
                                                          " Batch Size:\t\t{5:G}" + nwl + 
                                                          " Rate:\t\t\t  {6:0.#######}" + nwl;

        private readonly string stringLayer             = "Layer" + nwl;
        private readonly string stringNeurons           = "Neurons" + nwl;
        private readonly string stringWeights           = "Weights" + nwl;
        private readonly string stringBiases            = "Biases" + nwl;
        private readonly string stringTimings           = "Timings" + nwl;
        private readonly string stringFprop             = " fprop:  \t\t{0:D}/{1:D} ms";
        private readonly string stringBprop             = " bprop:  \t\t{0:D}/{1:D} ms" + nwl;
        private readonly string stringUpdate            = " update: \t\t{0:D}/{1:D} ms";
        private readonly string stringStdDevPositive    = " Std:     {0:N8}" + nwl;
        private readonly string stringStdDevNegative    = " Std:    {0:N8}" + nwl;
        private readonly string stringMeanPositive      = " Mean:    {0:N8}" + nwl;
        private readonly string stringMeanNegative      = " Mean:   {0:N8}" + nwl;
        private readonly string stringMminPositive      = " Min:     {0:N8}" + nwl;
        private readonly string stringMinNegative       = " Min:    {0:N8}" + nwl;
        private readonly string stringMaxPositive       = " Max:     {0:N8}" + nwl;
        private readonly string stringMaxNegative       = " Max:    {0:N8}" + nwl;

        private string? progressText = string.Empty;
        private string? layerInfo = string.Empty;
        private string? weightsMinMax = string.Empty;
        private string? label = string.Empty;
        private bool showProgress = false;
        private bool showSample = false;
        private ObservableCollection<DNNTrainingRate>? trainRates = new ObservableCollection<DNNTrainingRate>();
        private ObservableCollection<DNNTrainingStrategy>? trainingStrategies = new ObservableCollection<DNNTrainingStrategy>();
        private int selectedIndex = Settings.Default.SelectedIndex;
        private bool sgdr = false;
        private uint gotoEpoch = 1;
        private uint gotoCycle = 1;
        private int selectedCostIndex = 0;
        private ComboBox? optimizerComboBox;
        private ComboBox? costLayersComboBox;
        private ComboBox? dataProviderComboBox;
        private ComboBox? layersComboBox;
        private Button? refreshButton;
        private ComboBox? plotTypeComboBox;
        private CheckBox? disableLockingCheckBox;
        private Button? unlockAllButton;
        private Button? lockAllButton;
        private CheckBox? trainingPlotCheckBox;
        private Slider? pixelSizeSlider;
        private DNNOptimizers optimizer;
        private int refreshRate = 0;
        private int weightsSnapshotX = 0;
        private int weightsSnapshotY = 0;
        private bool showWeights = false;
        private bool showWeightsSnapshot = false;
        private bool showTrainingPlot = false;
        private ObservableCollection<DataPoint>? pointsTrain = new ObservableCollection<DataPoint>();
        private ObservableCollection<DataPoint>? pointsTest = new ObservableCollection<DataPoint>();
        private ObservableCollection<DNNTrainingResult>? selectedItems = Settings.Default.SelectedItems;
        private string? pointsTrainLabel = string.Empty;
        private string? pointsTestLabel = string.Empty;
        private PlotType currentPlotType;
        private LegendPosition currentLegendPosition;
        private PlotModel? plotModel;
        private Avalonia.Media.Imaging.WriteableBitmap? weightsSnapshot;
        private Avalonia.Media.Imaging.WriteableBitmap? inputSnapshot;
        private StringBuilder sb = new StringBuilder();
        public Timer? RefreshTimer;
        public TimeSpan EpochDuration { get; set; }
        public event EventHandler? Open;
        public event EventHandler? Save;
        public event EventHandler<int> RefreshRateChanged;

        public bool IsUpdating = false;
        public ReactiveCommand<SelectionChangedEventArgs, Unit> SelectionChangedCommand { get; }

        public void SelectionChanged(SelectionChangedEventArgs e)
        {
            if (!IsUpdating && TrainingLog != null)
            {
                if (e.AddedItems.Count > 0)
                {
                    foreach (var item in e.AddedItems)
                        if (item is DNNTrainingResult row)
                            SelectedItems?.Add(row);
                }
                if (e.RemovedItems.Count > 0)
                {
                    foreach (var item in e.RemovedItems)
                        if (item is DNNTrainingResult row)
                            SelectedItems?.Remove(row);
                }

                Settings.Default.SelectedItems = SelectedItems;
                Settings.Default.Save();

                e.Handled = true;
            }
        }

        public TrainPageViewModel(DNNModel model) : base(model)
        {
            showProgress = false;
            showSample = false;
            showWeights = false;
            showWeightsSnapshot = false;

            if (Model != null)
            {
                Model.NewEpoch += NewEpoch;
                Model.TrainProgress += TrainProgress;

                if (Settings.Default.SelectedItems == null)
                    Settings.Default.SelectedItems = new ObservableCollection<DNNTrainingResult>();

                SelectedItems = Settings.Default.SelectedItems;
            }

            SelectionChangedCommand = ReactiveCommand.Create<SelectionChangedEventArgs>(SelectionChanged);

            InitializeTrainingPlot();

            refreshRate = Settings.Default.RefreshInterval;
            sgdr = Settings.Default.SGDR;
            gotoEpoch = Settings.Default.GotoEpoch;
            gotoCycle = Settings.Default.GotoCycle;
            showTrainingPlot = Settings.Default.ShowTrainingPlot;
            currentPlotType = (PlotType)Settings.Default.PlotType;
            currentLegendPosition = currentPlotType == PlotType.Accuracy ? LegendPosition.BottomRight : LegendPosition.TopRight;
            optimizer = (DNNOptimizers)Settings.Default.Optimizer;
            
            AddCommandButtons();

            Modelhanged += TrainPageViewModel_ModelChanged;
            RefreshRateChanged += TrainPageViewModel_RefreshRateChanged;

            PixelSizeSlider_ValueChanged(this, null);
            LayersComboBox_SelectionChanged(this, null);
            RefreshTrainingPlot();
        }

        private void AddCommandButtons()
        {
            Button startButton = new Button
            {
                Name = "ButtonStart",
                Content = ApplicationHelper.LoadFromResource("Play.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(startButton, "Start Training");
            startButton.Click += StartButtonClick;

            Button stopButton = new Button
            {
                Name = "ButtonStop",
                Content = ApplicationHelper.LoadFromResource("Stop.png"),
                ClickMode = ClickMode.Release,
                IsVisible = false
            };
            ToolTip.SetTip(stopButton, "Stop Training");
            stopButton.Click += StopButtonClick;

            Button pauseButton = new Button
            {
                Name = "ButtonPause",
                Content = ApplicationHelper.LoadFromResource("Pause.png"),
                ClickMode = ClickMode.Release,
                IsVisible = false
            };
            ToolTip.SetTip(pauseButton, "Pause Training");
            pauseButton.Click += PauseButtonClick;

            Button editorButton = new Button
            {
                Name = "ButtonEditor",
                Content = ApplicationHelper.LoadFromResource("Collection.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(editorButton, "Training Scheme Editor");
            editorButton.Click += EditorButtonClick;

            Button strategiesButton = new Button
            {
                Name = "ButtonStrategies",
                Content = ApplicationHelper.LoadFromResource("Property.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(strategiesButton, "Training Strategies Editor");
            strategiesButton.Click += StrategyButtonClick;

            Button openButton = new Button
            {
                Name = "ButtonOpen",
                Content = ApplicationHelper.LoadFromResource("Open.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(openButton, "Load Model Weights");
            openButton.Click += OpenButtonClick;

            Button saveButton = new Button
            {
                Name = "ButtonSave",
                Content = ApplicationHelper.LoadFromResource("Save.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(saveButton, "Save Model Weights");
            saveButton.Click += SaveButtonClick;

            Button forgetButton = new Button
            {
                Name = "ButtonForgetWeights",
                Content = ApplicationHelper.LoadFromResource("Bolt.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(forgetButton, "Forget Model Weights");
            forgetButton.Click += ForgetButtonClick;

            Button clearButton = new Button
            {
                Name = "ButtonClearLog",
                Content = ApplicationHelper.LoadFromResource("ClearContents.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(clearButton, "Clear Log");
            clearButton.Click += ClearButtonClick;

            dataProviderComboBox = new ComboBox
            {
                Name = "ComboBoxDataSet",
                ItemsSource = Enum.GetValues<DNNDatasets>().Cast<Enum>().ToList(),
                SelectedIndex = (int)Dataset,
                IsEnabled = false
            };
            ToolTip.SetTip(dataProviderComboBox, "Dataset");

            optimizerComboBox = new ComboBox
            {
                Name = "ComboBoxOptimizers",
                ItemsSource = Enum.GetValues<DNNOptimizers>().Cast<Enum>().ToList(),
                IsEnabled = false
            };
            ToolTip.SetTip(optimizerComboBox, "Optimizer");
            Binding optBinding = new Binding
            {
                Source = this,
                Path = "Optimizer",
                Mode = BindingMode.TwoWay,
                Converter = new Converters.EnumConverter(),
                ConverterParameter = typeof(DNNOptimizers)
            };
            optimizerComboBox.Bind(ComboBox.SelectedValueProperty, optBinding);

            costLayersComboBox = new ComboBox
            {
                Name = "ComboBoxCostLayers"
            };
            costLayersComboBox.Items.Clear();
            if (Model?.CostLayers != null)
            {
                for (uint layer = 0u; layer < Model?.CostLayerCount; layer++)
                {
                    var item = new ComboBoxItem
                    {
                        Name = "CostLayer" + layer.ToString(),
                        Content = Model.CostLayers[layer].Name,
                        Tag = layer
                    };
                    costLayersComboBox.Items.Add(item);
                }
            }
            ToolTip.SetTip(costLayersComboBox, "Cost Layer");
            if (Model != null) 
                costLayersComboBox.SelectedIndex = (int)Model.CostIndex;
            selectedCostIndex = costLayersComboBox.SelectedIndex;
            costLayersComboBox.SelectionChanged += CostLayersComboBox_SelectionChanged;
            if (Model != null)
                costLayersComboBox.IsVisible = Model.CostLayerCount > 1;

            layersComboBox = new ComboBox { Name = "ComboBoxLayers" };
            layersComboBox.DataContext = Model;
            if (Model != null)
                layersComboBox.ItemsSource = Model.Layers;
            layersComboBox.ItemTemplate = new FuncDataTemplate<DNNLayerInfo>((value, namescope) => new TextBlock { [!TextBlock.TextProperty] = new Binding("Name"), [!TextBlock.FontWeightProperty] = new Binding { Path = "HasWeights", Mode = BindingMode.OneWay, Converter = new Converters.BoolToFontWeightConverter(), ConverterParameter = typeof(FontWeight) } });
            //layersComboBox.ItemTemplate = new FuncDataTemplate<DNNLayerInfo>((value, namescope) => new CheckBox {[!CheckBox.IsHitTestVisibleProperty] = new Binding { Path = "HasWeights", Mode = BindingMode.OneWay }, [!CheckBox.ContentProperty] = new Binding("Name"), [!CheckBox.FontWeightProperty] = new Binding {Path = "HasWeights", Mode = BindingMode.OneWay, Converter = new Converters.BoolToFontWeightConverter(), ConverterParameter = typeof(FontWeight) }});
            //layersComboBox.ItemTemplate = GetLockTemplate();
            //layersComboBox.SourceUpdated += LayersComboBox_SourceUpdated;
            //layersComboBox.IsSynchronizedWithCurrentItem = true;
            layersComboBox.SelectedIndex = Settings.Default.SelectedLayer;
            layersComboBox.SelectionChanged += LayersComboBox_SelectionChanged;
            ToolTip.SetTip(layersComboBox, "Layer");
            if (Model != null)
                Model.SelectedIndex = Settings.Default.SelectedLayer;

            disableLockingCheckBox = new CheckBox
            {
                Name = "CheckBoxDisableLocking",
                Content = ApplicationHelper.LoadFromResource("Key.png"),
                IsChecked = Settings.Default.DisableLocking
            };
            ToolTip.SetTip(disableLockingCheckBox, "Disable Locking");
            disableLockingCheckBox.IsCheckedChanged += DisableLockingCheckBox_IsCheckedChanged;

            unlockAllButton = new Button
            {
                Name = "UnlockAllButton",
                Content = ApplicationHelper.LoadFromResource("Unlock.png"),
                ClickMode = ClickMode.Release,
                IsVisible = !Settings.Default.DisableLocking && Model != null && Model.Layers != null && Model.Layers[Settings.Default.SelectedLayer].Lockable
            };
            ToolTip.SetTip(unlockAllButton, "Unlock All");
            unlockAllButton.Click += UnlockAll_Click;

            lockAllButton = new Button
            {
                Name = "LockAllButton",
                Content = ApplicationHelper.LoadFromResource("Lock.png"),
                ClickMode = ClickMode.Release,
                IsVisible = !Settings.Default.DisableLocking && Model != null && Model.Layers != null && Model.Layers[Settings.Default.SelectedLayer].Lockable
            };
            ToolTip.SetTip(lockAllButton, "Lock All");
            lockAllButton.Click += LockAll_Click;

            Button openLayerWeightsButton = new Button
            {
                Name = "ButtonOpenWeightsLayer",
                Content = ApplicationHelper.LoadFromResource("Open.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(openLayerWeightsButton, "Load Weights");
            openLayerWeightsButton.Click += OpenLayerWeightsButtonClick;

            Button saveLayerWeightsButton = new Button
            {
                Name = "ButtonSaveWeightsLayer",
                Content = ApplicationHelper.LoadFromResource("Save.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(saveLayerWeightsButton, "Save Weights");
            saveLayerWeightsButton.Click += SaveLayerWeightsButtonClick;

            Button forgetLayerWeightsButton = new Button
            {
                Name = "ButtonForgetWeightsLayer",
                Content = ApplicationHelper.LoadFromResource("LightningBolt.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(forgetLayerWeightsButton, "Forget Weights");
            forgetLayerWeightsButton.Click += ForgetLayerWeightsButtonClick;

            trainingPlotCheckBox = new CheckBox
            {
                Name = "CheckBoxTrainingPlot",
                Content = ApplicationHelper.LoadFromResource("PerformanceLog.png")
            };
            ToolTip.SetTip(trainingPlotCheckBox, "Training Plot");
            Binding tpBinding = new Binding
            {
                Source = this,
                Path = "ShowTrainingPlot",
                Mode = BindingMode.TwoWay,
                UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
            };
            trainingPlotCheckBox.Bind(CheckBox.IsCheckedProperty, tpBinding);
            trainingPlotCheckBox.IsCheckedChanged += TrainingPlotCheckBox_IsCheckedChanged;

            plotTypeComboBox = new ComboBox
            {
                Name = "ComboBoxPlotType",
                ItemsSource = Enum.GetValues(typeof(PlotType)).Cast<Enum>().ToList()
            };
            ToolTip.SetTip(plotTypeComboBox, "Plot Type");
            Binding binding = new Binding
            {
                Source = this,
                Path = "ShowTrainingPlot",
                Mode = BindingMode.TwoWay,
                UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
            };
            plotTypeComboBox.Bind(ComboBox.IsVisibleProperty, binding);
            plotTypeComboBox.SelectionChanged += PlotTypeComboBox_SelectionChanged;
            binding = new Binding
            {
                Source = this,
                Path = "CurrentPlotType",
                Mode = BindingMode.TwoWay,
                Converter = new Converters.EnumConverter(),
                ConverterParameter = typeof(PlotType),
                UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
            };
            plotTypeComboBox.Bind(ComboBox.SelectedValueProperty, binding);

            pixelSizeSlider = new Slider
            {
                Name = "PixelSizeSlider",
                Minimum = 1,
                Maximum = 8,
                LargeChange = 1,
                SmallChange = 1,
                Width = 106,
                Height = 24,
                Value = Settings.Default.PixelSize
            };
            ToolTip.SetTip(pixelSizeSlider, Math.Round(Settings.Default.PixelSize) == 1 ? "1 Pixel" : Math.Round(Settings.Default.PixelSize).ToString() + " Pixels");
            binding = new Binding
            {
                Source = this,
                Path = "ShowTrainingPlot",
                Mode = BindingMode.TwoWay,
                Converter = new Converters.InverseBooleanToVisibilityConverter(),
                UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
            };
            pixelSizeSlider.Bind(Slider.IsVisibleProperty, binding);
            pixelSizeSlider.ValueChanged += PixelSizeSlider_ValueChanged;

            refreshButton = new Button
            {
                Name = "ButtonRefresh",
                Content = ApplicationHelper.LoadFromResource("Refresh.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(refreshButton, "Refresh");
            refreshButton.Click += RefreshButtonClick;

            NumericUpDown refreshRateIntegerUpDown = new NumericUpDown
            {
                Name = "RefreshRate",
                HorizontalContentAlignment = Avalonia.Layout.HorizontalAlignment.Right,
                Minimum = 1,
                Maximum = 300,
                Increment = 1,
                ParsingNumberStyle = System.Globalization.NumberStyles.Integer,
                ClipValueToMinMax = true,
                Focusable = true
            };
            ToolTip.SetTip(refreshRateIntegerUpDown, "Refresh Rate/s");
            binding = new Binding
            {
                Source = this,
                Path = "RefreshRate",
                Mode = BindingMode.TwoWay
            };
            refreshRateIntegerUpDown.Bind(NumericUpDown.ValueProperty, binding);

            CommandToolBar.Add(startButton);                        // 0
            CommandToolBar.Add(stopButton);                         // 1
            CommandToolBar.Add(pauseButton);                        // 2
            CommandToolBar.Add(editorButton);                       // 3
            CommandToolBar.Add(strategiesButton);                   // 4
            CommandToolBar.Add(new Separator());                    // 5
            CommandToolBar.Add(openButton);                         // 6
            CommandToolBar.Add(saveButton);                         // 7
            CommandToolBar.Add(forgetButton);                       // 8
            CommandToolBar.Add(clearButton);                        // 9
            CommandToolBar.Add(new Separator());                    // 10
            CommandToolBar.Add(dataProviderComboBox);               // 11
            CommandToolBar.Add(optimizerComboBox);                  // 12
            CommandToolBar.Add(new Separator());                    // 13
            CommandToolBar.Add(costLayersComboBox);                 // 14
            CommandToolBar.Add(layersComboBox);                     // 15
            CommandToolBar.Add(disableLockingCheckBox);             // 16
            CommandToolBar.Add(unlockAllButton);                    // 17
            CommandToolBar.Add(lockAllButton);                      // 18
            CommandToolBar.Add(openLayerWeightsButton);             // 19
            CommandToolBar.Add(saveLayerWeightsButton);             // 20
            CommandToolBar.Add(forgetLayerWeightsButton);           // 21
            CommandToolBar.Add(new Separator());                    // 22
            CommandToolBar.Add(trainingPlotCheckBox);               // 23
            CommandToolBar.Add(plotTypeComboBox);                   // 24
            CommandToolBar.Add(pixelSizeSlider);                    // 25
            CommandToolBar.Add(new Separator());                    // 26
            CommandToolBar.Add(refreshButton);                      // 27
            CommandToolBar.Add(refreshRateIntegerUpDown);           // 28
        }
        
        private void TrainPageViewModel_RefreshRateChanged(object? sender, int e)
        {
            if (RefreshTimer != null)
                RefreshTimer.Interval = 1000 * e;
        }

        private async void TrainPageViewModel_ModelChanged(object? sender, EventArgs e)
        {
            showProgress = false;
            showSample = false;
            showWeights = false;
            showWeightsSnapshot = false;

            gotoEpoch = Settings.Default.GotoEpoch;
            showTrainingPlot = Settings.Default.ShowTrainingPlot;
            currentPlotType = (PlotType)Settings.Default.PlotType;
            currentLegendPosition = currentPlotType == PlotType.Accuracy ? LegendPosition.BottomRight : LegendPosition.TopRight;

            if (Model != null && Model.CostLayers != null)
            {
                Model.NewEpoch += NewEpoch;
                Model.TrainProgress += TrainProgress;
            

                costLayersComboBox?.Items.Clear();
                for (uint layer = 0u; layer < Model?.CostLayerCount; layer++)
                {
                    var item = new ComboBoxItem
                    {
                        Name = "CostLayer" + layer.ToString(),
                        Content = Model.CostLayers[layer].Name,
                        Tag = layer
                    };
                    costLayersComboBox?.Items.Add(item);
                }
            }
            if (Model != null && costLayersComboBox != null && layersComboBox != null)
            {
                costLayersComboBox.SelectedIndex = (int)Model.CostIndex;
                selectedCostIndex = costLayersComboBox.SelectedIndex;
                costLayersComboBox.IsEnabled = Model.CostLayerCount > 1;
                costLayersComboBox.IsVisible = Model.CostLayerCount > 1;
                layersComboBox.ItemsSource = Model.Layers;
                layersComboBox.SelectedIndex = 0;
                Model.SelectedIndex = 0;
            }

            SelectedItems = new ObservableCollection<DNNTrainingResult>();

            Settings.Default.SelectedLayer = 0;
            Settings.Default.Save();
            if (dataProviderComboBox  != null)
                dataProviderComboBox.SelectedIndex = (int)Dataset;

            LayersComboBox_SelectionChanged(sender, null);

            if (TrainingLog?.Count > 0)
            {
                var clear = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you want to clear the training log?", "Clear log?", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button1));
                if (clear == MessageBoxResult.Yes)
                     TrainingLog.Clear();
            }

            RefreshTrainingPlot();
        }

        private void NewEpoch(UInt Cycle, UInt Epoch, UInt TotalEpochs, UInt Optimizer, Float Beta2, Float Gamma, Float Eps, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, UInt Interpolation, Float Scaling, Float Rotation, Float Rate, UInt N, UInt D, UInt H, UInt W, UInt PadD, UInt PadH, UInt PadW, Float Momentum, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, UInt ElapsedNanoSecondes)
        {
            Dispatcher.UIThread.Invoke(() =>
            {
                if (Model != null && Model.CostLayers != null)
                {
                    var span = Model.Duration.Elapsed.Subtract(EpochDuration);
                    EpochDuration = Model.Duration.Elapsed;
                    for (UInt c = 0; c < Model.CostLayerCount; c++)
                    {
                        Model.UpdateCostInfo(c);
                        DNNCostLayer cost = Model.CostLayers[c];
                        TrainingLog?.Add(new DNNTrainingResult(Cycle, Epoch, cost.GroupIndex, c, cost.Name, N, D, H, W, PadD, PadH, PadW, (DNNOptimizers)Optimizer, Rate, Eps, Momentum, Beta2, Gamma, L2Penalty, Dropout, InputDropout, Cutout, CutMix, AutoAugment, HorizontalFlip, VerticalFlip, ColorCast, ColorAngle, Distortion, (DNNInterpolations)Interpolation, Scaling, Rotation, cost.AvgTrainLoss, cost.TrainErrors, cost.TrainErrorPercentage, cost.TrainAccuracy, cost.AvgTestLoss, cost.TestErrors, cost.TestErrorPercentage, cost.TestAccuracy, (Int64)span.TotalMilliseconds, span));
                    }
                    if (TrainingLog != null)
                        SelectedIndex = TrainingLog.Count - 1;

                    var epoch = "(" + Dataset.ToString().ToLower() + ")(" + ((DNNOptimizers)Optimizer).ToString().ToLower() + ")" + Epoch.ToString() + "-" + Cycle.ToString() + "-" + TrainErrors.ToString() + "-" + TestErrors.ToString();
                    var path = Path.Combine(DefinitionsDirectory, Settings.Default.ModelNameActive, epoch);
                    if (Directory.Exists(path))
                    {
                        try
                        {
                            File.WriteAllText(Path.Combine(path, "Program.cs"), Settings.Default.ScriptActive.Trim() == String.Empty ? Settings.Default.Script : Settings.Default.ScriptActive);
                        }
                        catch (Exception)
                        { 
                        }
                    }
                }
            }, DispatcherPriority.Send);

            RefreshTrainingPlot();

            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);
            GC.WaitForFullGCComplete();
        }

        private string FloatToString(Float number)
        {
            return number > 0 ? number.ToString() : "No";
        }

        private string BoolToString(bool value)
        {
            return value ? "Yes" : "No";
        }

        private string CutMixToString(bool cutMix)
        {
            return cutMix ? " CutMix:\t\t\t" : " Cutout:\t\t\t";
        }

        private void TrainProgress(DNNOptimizers Optim, UInt BatchSize, UInt Cycle, UInt TotalCycles, UInt Epoch, UInt TotalEpochs, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, DNNInterpolations Interpolation, Float Scaling, Float Rotation, UInt SampleIndex, Float Rate, Float Momentum, Float Beta2, Float Gamma, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, DNNStates State, DNNTaskStates TaskState)
        {
            Dispatcher.UIThread.Invoke(() =>
            {
                switch (State)
                {
                    case DNNStates.Training:
                        {
                            if (Optimizer != Optim && Model != null)
                            {
                                Optimizer = Optim;
                                Model.Optimizer = Optim;
                            }

                            switch (Model?.Optimizer)
                            {
                                case DNNOptimizers.AdaGrad:
                                    ProgressText = string.Format(stringTraining + " Dropout:\t\t   {7}" + nwl + CutMixToString(CutMix) + "{8}" + nwl + " Auto Augment:\t  {9}" + nwl + " Horizontal Flip:   {10}" + nwl + " Vertical Flip:\t {11}" + nwl + " Color Cast:\t\t{12}" + nwl + " Distortion:\t\t{13}" + nwl + " Loss:\t\t\t  {14:N7}" + nwl + " Errors:\t\t\t{15:G}" + nwl + " Error:\t\t\t {16:N2} %" + nwl + " Accuracy:\t\t  {17:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;

                                case DNNOptimizers.AdaDelta:
                                case DNNOptimizers.RMSProp:
                                    ProgressText = string.Format(stringTraining + " Momentum: \t\t {7:0.#######}" + nwl + " Dropout:\t\t   {8}" + nwl + CutMixToString(CutMix) + "{9}" + nwl + " Auto Augment:\t  {10}" + nwl + " Horizontal Flip:   {11}" + nwl + " Vertical Flip:\t {12}" + nwl + " Color Cast:\t\t{13}" + nwl + " Distortion:\t\t{14}" + nwl + " Loss:\t\t\t  {15:N7}" + nwl + " Errors:\t\t\t{16:G}" + nwl + " Error:\t\t\t {17:N2} %" + nwl + " Accuracy:\t\t  {18:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, Momentum, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;

                                case DNNOptimizers.AdaBoundW:
                                case DNNOptimizers.AdamW:
                                case DNNOptimizers.AmsBoundW:
                                case DNNOptimizers.AdamS:
                                    ProgressText = string.Format(stringTraining + " Momentum: \t\t {7:0.#######}" + nwl + " Beta2:\t\t\t {8:0.#######}" + nwl + " L2 Penalty:\t\t{9:0.#######}" + nwl + " Dropout:\t\t   {10}" + nwl + CutMixToString(CutMix) + "{11}" + nwl + " Auto Augment:\t  {12}" + nwl + " Horizontal Flip:   {13}" + nwl + " Vertical Flip:\t {14}" + nwl + " Color Cast:\t\t{15}" + nwl + " Distortion:\t\t{16}" + nwl + " Loss:\t\t\t  {17:N7}" + nwl + " Errors:\t\t\t{18:G}" + nwl + " Error:\t\t\t {19:N2} %" + nwl + " Accuracy:\t\t  {20:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, Momentum, Beta2, L2Penalty, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;

                                case DNNOptimizers.AdaBelief:
                                case DNNOptimizers.AdaBound:
                                case DNNOptimizers.Adam:
                                case DNNOptimizers.Adamax:
                                case DNNOptimizers.AmsBound:
                                case DNNOptimizers.DiffGrad:
                                    ProgressText = string.Format(stringTraining + " Momentum: \t\t {7:0.#######}" + nwl + " Beta2:\t\t\t {8:0.#######}" + nwl + " Dropout:\t\t   {9}" + nwl + CutMixToString(CutMix) + "{10}" + nwl + " Auto Augment:\t  {11}" + nwl + " Horizontal Flip:   {12}" + nwl + " Vertical Flip:\t {13}" + nwl + " Color Cast:\t\t{14}" + nwl + " Distortion:\t\t{15}" + nwl + " Loss:\t\t\t  {16:N7}" + nwl + " Errors:\t\t\t{17:G}" + nwl + " Error:\t\t\t {18:N2} %" + nwl + " Accuracy:\t\t  {19:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, Momentum, Beta2, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;

                                case DNNOptimizers.SGD:
                                    ProgressText = string.Format(stringTraining + " L2 Penalty:\t\t{7:0.#######}" + nwl + " Dropout:\t\t   {8}" + nwl + CutMixToString(CutMix) + "{9}" + nwl + " Auto Augment:\t  {10}" + nwl + " Horizontal Flip:   {11}" + nwl + " Vertical Flip:\t {12}" + nwl + " Color Cast:\t\t{13}" + nwl + " Distortion:\t\t{14}" + nwl + " Loss:\t\t\t  {15:N7}" + nwl + " Errors:\t\t\t{16:G}" + nwl + " Error:\t\t\t {17:N2} %" + nwl + " Accuracy:\t\t  {18:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, L2Penalty, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;

                                case DNNOptimizers.NAG:
                                case DNNOptimizers.SGDMomentum:
                                case DNNOptimizers.SGDW:
                                    ProgressText = string.Format(stringTraining + " Momentum:\t\t  {7:0.#######}" + nwl + " L2 Penalty:\t\t{8:0.#######}" + nwl + " Dropout:\t\t   {9}" + nwl + CutMixToString(CutMix) + "{10}" + nwl + " Auto Augment:\t  {11}" + nwl + " Horizontal Flip:   {12}" + nwl + " Vertical Flip:\t {13}" + nwl + " Color Cast:\t\t{14}" + nwl + " Distortion:\t\t{15}" + nwl + " Loss:\t\t\t  {16:N7}" + nwl + " Errors:\t\t\t{17:G}" + nwl + " Error:\t\t\t {18:N2} %" + nwl + " Accuracy:\t\t  {19:N2} %", SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, Rate, Momentum, L2Penalty, FloatToString(Dropout), FloatToString(Cutout), FloatToString(AutoAugment), BoolToString(HorizontalFlip), BoolToString(VerticalFlip), FloatToString(ColorCast), FloatToString(Distortion), AvgTrainLoss, TrainErrors, TrainErrorPercentage, 100 - TrainErrorPercentage);
                                    break;
                            }
                        }
                        break;

                    case DNNStates.Testing:
                        {
                            if (Model != null)
                                ProgressText = string.Format(stringTesting, SampleIndex, Cycle, TotalCycles, Epoch, TotalEpochs, Model.BatchSize, AvgTestLoss, TestErrors, TestErrorPercentage, 100 - TestErrorPercentage);
                        }
                        break;

                    case DNNStates.SaveWeights:
                        ProgressText = "Saving weights" + nwl;
                        break;

                    case DNNStates.Completed:
                        {
                            if (RefreshTimer != null)
                            {
                                RefreshTimer.Stop();
                                RefreshTimer.Elapsed -= new ElapsedEventHandler(RefreshTimer_Elapsed);
                                RefreshTimer.Dispose();
                            }

                            if (Model != null)
                            {
                                Model.Stop();

                                ProgressText = string.Empty;

                                ToolTip.SetTip(CommandToolBar[0], "Start Training");
                                CommandToolBar[0].IsVisible = true;
                                CommandToolBar[1].IsVisible = false;
                                CommandToolBar[2].IsVisible = false;

                                CommandToolBar[5].IsVisible = true;
                                CommandToolBar[6].IsVisible = true;
                                CommandToolBar[7].IsVisible = true;

                                if (layersComboBox != null && Model.Layers != null && (Model.Layers[layersComboBox.SelectedIndex].WeightCount > 0 || Model.Layers[layersComboBox.SelectedIndex].IsNormLayer))
                                {
                                    CommandToolBar[16].IsVisible = !Settings.Default.DisableLocking;
                                    CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
                                    CommandToolBar[18].IsVisible = true;
                                    CommandToolBar[19].IsVisible = true;
                                }
                                else
                                {
                                    CommandToolBar[16].IsVisible = false;
                                    CommandToolBar[17].IsVisible = false;
                                    CommandToolBar[18].IsVisible = false;
                                    CommandToolBar[19].IsVisible = false;
                                }
                            }
                            ShowProgress = false;
                        }
                        break;
                }
            });
        }

        public void OnDisableLockingChanged(object? sender, RoutedEventArgs e)
        {
            if (disableLockingCheckBox !=  null)
                disableLockingCheckBox.IsChecked = Settings.Default.DisableLocking;
        }

        private void DisableLockingCheckBox_IsCheckedChanged(object? sender, RoutedEventArgs e)
        {
            if (disableLockingCheckBox != null && disableLockingCheckBox.IsChecked.HasValue)
            {
                Settings.Default.DisableLocking = disableLockingCheckBox.IsChecked.Value;
                Settings.Default.Save();

                Model?.SetDisableLocking(Settings.Default.DisableLocking);

                if (unlockAllButton != null)
                    unlockAllButton.IsVisible = !Settings.Default.DisableLocking;
                if (lockAllButton != null)
                    lockAllButton.IsVisible = !Settings.Default.DisableLocking;

                //layersComboBox.ItemTemplate = GetLockTemplate();

                //int index = layersComboBox.SelectedIndex;
                //if (index > 0)
                //{
                //    layersComboBox.SelectedIndex = index - 1;
                //    layersComboBox.SelectedIndex = index;
                //}
                //else
                //    if (Model?.LayerCount > (ulong)(index + 1))
                //{
                //    layersComboBox.SelectedIndex = index + 1;
                //    layersComboBox.SelectedIndex = index;
                //}
            }
        }

        private void UnlockAll_Click(object? sender, RoutedEventArgs e)
        {
            Model?.SetLocked(false);
        }

        private void LockAll_Click(object? sender, RoutedEventArgs e)
        {
            Model?.SetLocked(true);
        }

        //static DataTemplate GetLockTemplate()
        //{
        //    DataTemplate checkBoxLayout = new DataTemplate
        //    {
        //        DataType = typeof(DNNLayerInfo)
        //    };
        //    // set up the StackPanel
        //    var panelFactory = new FrameworkElementFactory(typeof(StackPanel))
        //    {
        //        Name = "myComboFactory"
        //    };
        //    panelFactory.SetValue(StackPanel.OrientationProperty, Orientation.Horizontal);

        //    FrameworkElementFactory contentFactory;
        //    var color = System.Windows.Media.Color.FromArgb(255, 215, 199, 215);
        //    var brush = new System.Windows.Media.SolidColorBrush(color);
        //    brush.Freeze();
        //    if (!Settings.Default.DisableLocking)
        //    {

        //        //set up the CheckBox
        //        contentFactory = new FrameworkElementFactory(typeof(CheckBox));
        //        contentFactory.SetBinding(CheckBox.ContentProperty, new Binding("Name"));
        //        contentFactory.SetValue(Control.ForegroundProperty, brush);

        //        Binding bindingIsChecked = new Binding("LockUpdate")
        //        {
        //            Mode = BindingMode.TwoWay,
        //            UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged,
        //            NotifyOnSourceUpdated = true
        //        };
        //        contentFactory.SetBinding(CheckBox.IsCheckedProperty, bindingIsChecked);
        //        contentFactory.SetBinding(CheckBox.IsEnabledProperty, new Binding("Lockable"));
        //    }
        //    else
        //    {
        //        contentFactory = new FrameworkElementFactory(typeof(TextBlock));
        //        contentFactory.SetBinding(TextBlock.TextProperty, new Binding("Name"));
        //        contentFactory.SetValue(Control.ForegroundProperty, brush);
        //    }

        //    Binding bindingFontWeights = new Binding("Lockable");
        //    Converters.BoolToStringConverter converter = new Converters.BoolToStringConverter
        //    {
        //        TrueValue = System.Windows.FontWeights.ExtraBold,
        //        FalseValue = System.Windows.FontWeights.Normal
        //    };
        //    bindingFontWeights.Converter = converter;
        //    contentFactory.SetBinding(CheckBox.FontWeightProperty, bindingFontWeights);
        //    panelFactory.AppendChild(contentFactory);
        //    checkBoxLayout.VisualTree = panelFactory;

        //    return checkBoxLayout;
        //}

        //private void LayersComboBox_SourceUpdated(object? sender, DataTransferEventArgs e)
        //{
        //    if (e.OriginalSource is CheckBox cb)
        //    {
        //        if (cb.IsEnabled)
        //        {
        //            var layer = Model?.Layers.FirstOrDefault(i => i.Name == cb.Content as String);
        //            if (layer != null && layer.LockUpdate != null) 
        //                Model?.SetLayerLocked(layer.LayerIndex, layer.LockUpdate.Value);

        //            e.Handled = true;
        //        }
        //    }
        //}
       
        private void PixelSizeSlider_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs? e)
        {

            int temp = e != null ? (int)Math.Round(e.NewValue) : (int)Settings.Default.PixelSize;
            
            if (pixelSizeSlider != null)
            {
                if (temp == 1)
                    ToolTip.SetTip(pixelSizeSlider, "1 Pixel");
                else
                    ToolTip.SetTip(pixelSizeSlider, temp.ToString() + " Pixels");
            }

            Settings.Default.PixelSize = temp;
            Settings.Default.Save();

            if (Model != null)
                Model.BlockSize = (ulong)temp;

            Dispatcher.UIThread.Invoke(() =>
            {
                if (Model != null && Model.Layers != null && layersComboBox?.SelectedIndex >= 0)
                {
                    var index = layersComboBox.SelectedIndex;
                    if (index < (int)Model.LayerCount)
                    {
                        Settings.Default.SelectedLayer = index;
                        Settings.Default.Save();
                        Model.SelectedIndex = index;

                        ShowSample = Model.TaskState == DNNTaskStates.Running;
                        ShowWeights = Model.Layers[index].WeightCount > 0 || Settings.Default.Timings;
                        ShowWeightsSnapshot = (Model.Layers[index].IsNormLayer && Model.Layers[index].Scaling) || Model.Layers[index].LayerType == DNNLayerTypes.DepthwiseConvolution || Model.Layers[index].LayerType == DNNLayerTypes.ConvolutionTranspose || Model.Layers[index].LayerType == DNNLayerTypes.Convolution || Model.Layers[index].LayerType == DNNLayerTypes.Dense || (Model.Layers[index].LayerType == DNNLayerTypes.Activation && Model.Layers[index].WeightCount > 0);

                        if (index == 0)
                            Model.UpdateLayerInfo((ulong)index, ShowSample);
                        else
                            Model.UpdateLayerInfo((ulong)index, ShowWeightsSnapshot);

                        WeightsSnapshotX = Model.Layers[index].WeightsSnapshotX;
                        WeightsSnapshotY = Model.Layers[index].WeightsSnapshotY;
                        WeightsSnapshot = Model.Layers[index].WeightsSnapshot;
                    }
                }
            });
        }

        private void TrainingPlotCheckBox_IsCheckedChanged(object? sender, RoutedEventArgs e)
        {
            Settings.Default.ShowTrainingPlot = trainingPlotCheckBox?.IsChecked ?? false;
            Settings.Default.Save();
        }

        private void PlotTypeComboBox_SelectionChanged(object? sender, SelectionChangedEventArgs e)
        {
            if (plotTypeComboBox != null)
            {
                CurrentPlotType = (PlotType)plotTypeComboBox.SelectedIndex;
                Settings.Default.PlotType = (uint)plotTypeComboBox.SelectedIndex;
                Settings.Default.Save();
            }
            
            RefreshTrainingPlot();
        }
               
        public void CostLayersComboBox_SelectionChanged(object? sender, SelectionChangedEventArgs? e)
        {
            if (costLayersComboBox?.SelectedIndex >= 0)
            {
                SelectedCostIndex = costLayersComboBox.SelectedIndex;
                Model?.SetCostIndex((uint)SelectedCostIndex);
            }
        }

        public void RefreshTrainingPlot()
        {
            Dispatcher.UIThread.Invoke(() =>
            {
                PointsTrain?.Clear();
                PointsTest?.Clear();

                if (TrainingLog != null)
                {
                    switch (CurrentPlotType)
                    {
                        case PlotType.Accuracy:
                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTrain?.Add(new DataPoint(result.Epoch, result.TrainAccuracy));

                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTest?.Add(new DataPoint(result.Epoch, result.TestAccuracy));

                            PointsTrainLabel = "Train Accuracy %";
                            PointsTestLabel = "Test Accuracy %";
                            break;

                        case PlotType.Error:
                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTrain?.Add(new DataPoint(result.Epoch, result.TrainErrorPercentage));

                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTest?.Add(new DataPoint(result.Epoch, result.TestErrorPercentage));

                            PointsTrainLabel = "Train Error %";
                            PointsTestLabel = "Test Error %";
                            break;

                        case PlotType.Loss:
                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTrain?.Add(new DataPoint(result.Epoch, result.AvgTrainLoss));

                            foreach (DNNTrainingResult result in TrainingLog)
                                if ((int)result.CostIndex == SelectedCostIndex)
                                    PointsTest?.Add(new DataPoint(result.Epoch, result.AvgTestLoss));

                            PointsTrainLabel = "Avg Train Loss";
                            PointsTestLabel = "Avg Test Loss";
                            break;
                    }
                }

                if (plotModel != null)
                {
                    plotModel.Series[0].Title = PointsTrainLabel;
                    plotModel.Series[1].Title = PointsTestLabel;
                }

                this.RaisePropertyChanged(nameof(PointsTrain));
                this.RaisePropertyChanged(nameof(PointsTest));

                plotModel?.InvalidatePlot(true);
                //this.RaisePropertyChanged(nameof(PlotModel));
            }, DispatcherPriority.Render);
        }

        private void InitializeTrainingPlot()
        {
            Dispatcher.UIThread.Invoke(() =>
            {
                plotModel = new PlotModel();
                var laLeft = new LinearAxis
                {
                    Title = "",
                    Position = AxisPosition.Left,
                    MajorGridlineStyle = LineStyle.Solid,
                    MinorGridlineStyle = LineStyle.Dot,
                    IsAxisVisible = true,
                    IsPanEnabled = false,
                    IsZoomEnabled = false
                };
                var laBottomn = new LinearAxis
                {
                    Title = "Epochs",
                    TitleFontSize = 14,
                    Position = AxisPosition.Bottom,
                    MajorGridlineStyle = LineStyle.Solid,
                    MinorGridlineStyle = LineStyle.Dot,
                    IsAxisVisible = true,
                    IsPanEnabled = false,
                    IsZoomEnabled = false
                };
                plotModel.Axes.Add(laLeft);
                plotModel.Axes.Add(laBottomn);
                pointsTrain = new ObservableCollection<DataPoint>();
                pointsTest = new ObservableCollection<DataPoint>();
                var lsTrain = new OxyPlot.Series.LineSeries
                {
                    ItemsSource = PointsTrain,
                    Title = PointsTrainLabel,
                    Color = OxyColor.FromRgb(237, 125, 49)
                };
                var lsTest = new OxyPlot.Series.LineSeries
                {
                    ItemsSource = PointsTest,
                    Title = PointsTestLabel,
                    Color = OxyColor.FromRgb(91, 155, 213)
                };
                plotModel.Series.Add(lsTrain);
                plotModel.Series.Add(lsTest);
                var legend = new Legend();
                legend.LegendFont = "Consolas";
                legend.LegendPosition = CurrentLegendPosition;
                legend.LegendTitleFontSize = 16;
                legend.LegendFontSize = 16;
                legend.LegendPosition = LegendPosition.RightBottom;
                plotModel.Legends.Add(legend);
                plotModel.TextColor = OxyColor.FromRgb(255, 255, 255);
                //this.RaisePropertyChanged(nameof(PlotModel));
            }, DispatcherPriority.Render);
        }

        public PlotModel? PlotModel
        {
            get => plotModel;
            set => this.RaiseAndSetIfChanged(ref plotModel, value);
        }

        public bool SGDR
        {
            get => sgdr;
            set
            {
                if (sgdr == value)
                    return;

                this.RaiseAndSetIfChanged(ref sgdr, value);
                Settings.Default.SGDR = sgdr;
                Settings.Default.Save();
            }
        }

        public uint GotoEpoch
        {
            get => gotoEpoch;
            set
            {
                if (gotoEpoch == value)
                    return;

                this.RaiseAndSetIfChanged(ref gotoEpoch, value);
                Settings.Default.GotoEpoch = gotoEpoch;
                Settings.Default.Save();
            }
        }

        public uint GotoCycle
        {
            get => gotoCycle;
            set
            {
                if (gotoCycle == value)
                    return;

                this.RaiseAndSetIfChanged(ref gotoCycle, value);
                Settings.Default.GotoCycle = gotoCycle;
                Settings.Default.Save();
            }
        }

        public LegendPosition CurrentLegendPosition
        {
            get => currentLegendPosition;
            set => this.RaiseAndSetIfChanged(ref currentLegendPosition, value);
        }

        public ObservableCollection<DataPoint>? PointsTrain
        {
            get => pointsTrain;
            set => this.RaiseAndSetIfChanged(ref pointsTrain, value);
        }

        public ObservableCollection<DataPoint>? PointsTest
        {
            get => pointsTest;
            set => this.RaiseAndSetIfChanged(ref pointsTest, value);
        }

        public string? PointsTrainLabel
        {
            get => pointsTrainLabel;
            set => this.RaiseAndSetIfChanged(ref pointsTrainLabel, value);
        }

        public string? PointsTestLabel
        {
            get => pointsTestLabel;
            set => this.RaiseAndSetIfChanged(ref pointsTestLabel, value);
        }

        public PlotType CurrentPlotType
        {
            get => currentPlotType;
            set
            {
                if (currentPlotType == value)
                    return;

                this.RaiseAndSetIfChanged(ref currentPlotType, value);
                
                switch (currentPlotType)
                {
                    case PlotType.Accuracy:
                        CurrentLegendPosition = LegendPosition.BottomRight;
                        break;
                    case PlotType.Error:
                    case PlotType.Loss:
                        CurrentLegendPosition = LegendPosition.TopRight;
                        break;
                }
            }
        }

        public string? ProgressText
        {
            get => progressText;
            set => this.RaiseAndSetIfChanged(ref progressText, value);
        }

        public bool ShowProgress
        {
            get => showProgress; 
            set => this.RaiseAndSetIfChanged(ref showProgress, value);
        }

        public string? LayerInfo
        {
            get => layerInfo;
            set => this.RaiseAndSetIfChanged(ref layerInfo, value);
        }

        public string? WeightsMinMax
        {
            get => weightsMinMax;
            set => this.RaiseAndSetIfChanged(ref weightsMinMax, value);
        }

        public int WeightsSnapshotX
        {
            get => weightsSnapshotX;
            set => this.RaiseAndSetIfChanged(ref weightsSnapshotX, value);
        }

        public int WeightsSnapshotY
        {
            get => weightsSnapshotY;
            set => this.RaiseAndSetIfChanged(ref weightsSnapshotY, value);
        }

        public string? Label
        {
            get => label;
            set => this.RaiseAndSetIfChanged(ref label, value);
        }

        public bool ShowSample
        {
            get => showSample;
            set => this.RaiseAndSetIfChanged(ref showSample, value);
        }

        public bool ShowWeights
        {
            get => showWeights;
            set => this.RaiseAndSetIfChanged(ref showWeights, value);
        }

        public bool ShowWeightsSnapshot
        {
            get => showWeightsSnapshot;
            set => this.RaiseAndSetIfChanged(ref showWeightsSnapshot, value);
        }

        public bool ShowTrainingPlot
        {
            get => showTrainingPlot;
            set => this.RaiseAndSetIfChanged(ref showTrainingPlot, value);
        }

        public Avalonia.Media.Imaging.WriteableBitmap? WeightsSnapshot
        {
            get => weightsSnapshot;
            set => this.RaiseAndSetIfChanged(ref weightsSnapshot, value);
        }

        public Avalonia.Media.Imaging.WriteableBitmap? InputSnapshot
        {
            get => inputSnapshot;
            set => this.RaiseAndSetIfChanged(ref inputSnapshot, value);
        }

        public DNNTrainingRate TrainRate
        {
            get => Settings.Default.TraininingRate ?? new DNNTrainingRate(DNNOptimizers.NAG, 0.9f, 0.999f, 0.0005f, 0, 1E-08f, 128, 1, 32, 32, 0, 4, 4, 1, 200, 1, 0.05f, 0.0001f, 0.1f, 0.003f, 1, 1, false, false, 0, 0, false, 0, 0, 0, 0, DNNInterpolations.Cubic, 10, 12);
            private set
            {
                if (value == Settings.Default.TraininingRate)
                    return;

                Settings.Default.TraininingRate = value;
                this.RaisePropertyChanged(nameof(TrainRate));
            }
        }

        public ObservableCollection<DNNTrainingRate>? TrainRates
        {
            get => trainRates;
            private set => this.RaiseAndSetIfChanged(ref trainRates, value);
        }

        public ObservableCollection<DNNTrainingStrategy>? TrainingStrategies
        {
            get => trainingStrategies;
            set => this.RaiseAndSetIfChanged(ref trainingStrategies, value);
        }

        public ObservableCollection<DNNTrainingResult>? SelectedItems
        {
            get => selectedItems;
            set 
            {
                if (value != selectedItems)
                {
                    Settings.Default.SelectedItems = value;
                    this.RaiseAndSetIfChanged(ref selectedItems, value);
                }
            }
        }

        public ObservableCollection<DNNTrainingResult>? TrainingLog
        {
            get
            {
                if (Settings.Default.TrainingLog == null)
                    Settings.Default.TrainingLog = new ObservableCollection<DNNTrainingResult>();

                return Settings.Default.TrainingLog;
            }

            set
            {
                if (value == Settings.Default.TrainingLog)
                    return;

                Settings.Default.TrainingLog = value;
                this.RaisePropertyChanged(nameof(TrainingLog));
            }
        }

        public int SelectedCostIndex
        {
            get => selectedCostIndex;
            set 
            {
                if (value == selectedCostIndex)
                    return;

                this.RaiseAndSetIfChanged(ref selectedCostIndex, value);
                RefreshTrainingPlot();
            }
        }

        public int SelectedIndex
        {
            get => selectedIndex;
            set 
            {
                if (selectedIndex != value)
                {
                    Settings.Default.SelectedIndex = value;
                    this.RaiseAndSetIfChanged(ref selectedIndex, value);
                }
            }
        }

        public DNNOptimizers Optimizer
        {
            get => optimizer;
            set
            {
                if (value == optimizer)
                    return;

                this.RaiseAndSetIfChanged(ref optimizer, value);
                Settings.Default.Optimizer = (int)optimizer;
                Settings.Default.Save();
            }
        }

        public override string DisplayName => "Train";

        public int RefreshRate
        {
            get => refreshRate;
            set
            {
                if (value == refreshRate)
                    return;

                this.RaiseAndSetIfChanged(ref refreshRate, value);

                Settings.Default.RefreshInterval = refreshRate;
                Settings.Default.Save();
                EventHandler<int> handler = RefreshRateChanged;
                handler.Invoke(this, refreshRate);
            }
        }

        public override void Reset()
        {
            if (TrainingLog != null)
            {
                TrainingLog.Clear();
                Model?.ClearLog();
            }
            SelectedIndex = -1;
            ProgressText = String.Empty;
            Label = String.Empty;
            RefreshTrainingPlot();
        }

        private void RefreshTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            LayersComboBox_SelectionChanged(sender, null);
        }

        private void StartButtonClick(object? sender, RoutedEventArgs e)
        {
            Dispatcher.UIThread.Post(async () =>
            {
                if (Model?.TaskState == DNNTaskStates.Running)
                {
                    await MessageBox.Show("You must stop testing first.", "Information", MessageBoxButtons.OK);
                    return;
                }

                if (Model?.TaskState == DNNTaskStates.Stopped)
                {
                    if (App.MainWindow != null)
                    {
                        var dialog = new TrainParameters
                        {
                            Model = this.Model,
                            Path = DefinitionsDirectory,
                            IsEnabled = true,
                            Rate = TrainRate,
                            tpvm = this,
                        };

                        await dialog.ShowDialog(App.MainWindow);

                        if (dialog.DialogResult)
                        {
                            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, true);

                            TrainRate = dialog.Rate;

                            if (SGDR)
                                Model.AddTrainingRateSGDR(TrainRate, true, GotoEpoch, GotoCycle, Model.TrainingSamples);
                            else
                                Model.AddTrainingRate(TrainRate, true, GotoEpoch, Model.TrainingSamples);

                            Model.SetOptimizer(TrainRate.Optimizer);
                            Model.Optimizer = TrainRate.Optimizer;
                            Optimizer = TrainRate.Optimizer;

                            EpochDuration = TimeSpan.Zero;

                            RefreshTimer = new Timer(1000 * Settings.Default.RefreshInterval);
                            RefreshTimer.Elapsed += new ElapsedEventHandler(RefreshTimer_Elapsed);

                            Model.SetCostIndex((uint)SelectedCostIndex);
                            Model.Start(true);
                            RefreshTimer.Start();
                            CommandToolBar[0].IsVisible = false;
                            CommandToolBar[1].IsVisible = true;
                            CommandToolBar[2].IsVisible = true;

                            CommandToolBar[6].IsVisible = false;
                            CommandToolBar[7].IsVisible = true;
                            CommandToolBar[8].IsVisible = false;

                            CommandToolBar[17].IsVisible = false;
                            CommandToolBar[18].IsVisible = false;
                            CommandToolBar[19].IsVisible = false;
                            CommandToolBar[20].IsVisible = false;
                            CommandToolBar[21].IsVisible = false;

                            if (layersComboBox != null && Model.Layers != null && Model.Layers[layersComboBox.SelectedIndex].WeightCount > 0)
                            {
                                if ((Model.Layers[layersComboBox.SelectedIndex].IsNormLayer && Model.Layers[layersComboBox.SelectedIndex].Scaling) || !Model.Layers[layersComboBox.SelectedIndex].IsNormLayer)
                                {
                                    CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
                                    CommandToolBar[18].IsVisible = !Settings.Default.DisableLocking;
                                    CommandToolBar[20].IsVisible = true;
                                }
                            }

                            ShowProgress = true;
                        }
                    }
                }
                else
                {
                    if (Model?.TaskState == DNNTaskStates.Paused)
                    {
                        Model.Resume();
                        CommandToolBar[0].IsVisible = false;
                        CommandToolBar[1].IsVisible = true;
                        CommandToolBar[2].IsVisible = true;

                        CommandToolBar[6].IsVisible = false;
                        CommandToolBar[7].IsVisible = true;
                        CommandToolBar[8].IsVisible = false;
                    }
                }
            }, DispatcherPriority.Normal);
        }

        private async void StopButtonClick(object? sender, RoutedEventArgs e)
        {
            if (Model != null && Model?.TaskState != DNNTaskStates.Stopped)
            {
                var stop = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to stop?", "Stop Training", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
                if (stop == MessageBoxResult.Yes)
                {
                    if (RefreshTimer != null) 
                    {
                        RefreshTimer.Stop();
                        RefreshTimer.Elapsed -= new ElapsedEventHandler(RefreshTimer_Elapsed);
                        RefreshTimer.Dispose();
                    }

                    Model?.Stop();

                    ToolTip.SetTip(CommandToolBar[0], "Start Training");
                    CommandToolBar[0].IsVisible = true;
                    CommandToolBar[1].IsVisible = false;
                    CommandToolBar[2].IsVisible = false;

                    CommandToolBar[6].IsVisible = true;
                    CommandToolBar[7].IsVisible = true;
                    CommandToolBar[8].IsVisible = true;

                    CommandToolBar[17].IsVisible = false;
                    CommandToolBar[18].IsVisible = false;
                    CommandToolBar[19].IsVisible = false;
                    CommandToolBar[20].IsVisible = false;
                    CommandToolBar[21].IsVisible = false;

                    if (layersComboBox != null && Model?.Layers != null && Model?.Layers[layersComboBox.SelectedIndex].WeightCount > 0)
                    {
                        if ((Model.Layers[layersComboBox.SelectedIndex].IsNormLayer && Model.Layers[layersComboBox.SelectedIndex].Scaling) || !Model.Layers[layersComboBox.SelectedIndex].IsNormLayer)
                        {
                            CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
                            CommandToolBar[18].IsVisible = !Settings.Default.DisableLocking;
                            CommandToolBar[19].IsVisible = true;
                            CommandToolBar[20].IsVisible = true;
                            CommandToolBar[21].IsVisible = true;
                        }
                    }

                    ShowProgress = false;
                }
            }
        }

        private void PauseButtonClick(object? sender, RoutedEventArgs e)
        {
            Dispatcher.UIThread.Post(() =>
            {
                if (Model?.TaskState == DNNTaskStates.Running)
                {
                    Model.Pause();
                    
                    ToolTip.SetTip(CommandToolBar[0], "Resume Training");
                    CommandToolBar[0].IsVisible = true;
                    CommandToolBar[1].IsVisible = true;
                    CommandToolBar[2].IsVisible = false;

                    CommandToolBar[6].IsVisible = false;
                    CommandToolBar[7].IsVisible = true;
                    CommandToolBar[8].IsVisible = false;
                }
            }, DispatcherPriority.Normal);
        }

        private void OpenButtonClick(object? sender, RoutedEventArgs e)
        {
            Open?.Invoke(this,  EventArgs.Empty);
        }

        private void SaveButtonClick(object? sender, RoutedEventArgs e)
        {
            Save?.Invoke(this, EventArgs.Empty);
        }

        private void EditorButtonClick(object? sender, RoutedEventArgs e)
        {
            //if (Model?.TaskState == DNNTaskStates.Stopped)
            //{
            //    if (Settings.Default.TrainingRates == null)
            //        Settings.Default.TrainingRates = new ObservableCollection<DNNTrainingRate> { TrainRate };
                      
            //    TrainRates = Settings.Default.TrainingRates;
                
            //    TrainingSchemeEditor dialog = new TrainingSchemeEditor { Path = StorageDirectory };
                
            //    dialog.tpvm = this;
            //    dialog.DataContext = this;
            //    dialog.buttonTrain.IsEnabled = true;
            //    dialog.Owner = Application.Current.MainWindow;
            //    dialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;

            //    if (dialog.ShowDialog() ?? false)
            //    {
            //        bool first = true;
            //        foreach (DNNTrainingRate rate in TrainRates)
            //        {
            //            if (SGDR)
            //                Model.AddTrainingRateSGDR(rate, first, GotoEpoch, GotoCycle, Model.TrainingSamples);
            //            else
            //                Model.AddTrainingRate(rate, first, GotoEpoch, Model.TrainingSamples);

            //            first = false;
            //        }

            //        EpochDuration = TimeSpan.Zero;

            //        RefreshTimer = new Timer(1000 * Settings.Default.RefreshInterval.Value);
            //        RefreshTimer.Elapsed += new ElapsedEventHandler(RefreshTimer_Elapsed);

            //        Model.SetOptimizer(TrainRates[0].Optimizer);
            //        Model.Optimizer = TrainRates[0].Optimizer;
            //        Optimizer = TrainRates[0].Optimizer;

            //        Model.Start(true);
            //        RefreshTimer.Start();
            //        CommandToolBar[0].IsVisible = false;
            //        CommandToolBar[1].IsVisible = true;
            //        CommandToolBar[2].IsVisible = true;

            //        CommandToolBar[6].IsVisible = false;
            //        CommandToolBar[7].IsVisible = true;
            //        CommandToolBar[8].IsVisible = false;
                  
            //        if (layersComboBox.SelectedIndex >= 0 && Model.Layers[layersComboBox.SelectedIndex].WeightCount > 0)
            //        {
            //            DNNLayerInfo info = Model.Layers[layersComboBox.SelectedIndex];
            //            if (info.IsNormLayer)
            //            {
            //                if (info.Scaling)
            //                {
            //                    CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
            //                    CommandToolBar[18].IsVisible = !Settings.Default.DisableLocking;
            //                    CommandToolBar[19].IsVisible = false;
            //                    CommandToolBar[20].IsVisible = true;
            //                    CommandToolBar[21].IsVisible = true;
            //                }
            //                else
            //                {
            //                    CommandToolBar[17].IsVisible = false;
            //                    CommandToolBar[18].IsVisible = false;
            //                    CommandToolBar[19].IsVisible = false;
            //                    CommandToolBar[20].IsVisible = false;
            //                    CommandToolBar[21].IsVisible = true;
            //                }
            //            }
            //            else
            //            {
            //                CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
            //                CommandToolBar[18].IsVisible = !Settings.Default.DisableLocking;
            //                CommandToolBar[19].IsVisible = false;
            //                CommandToolBar[20].IsVisible = true;
            //                CommandToolBar[21].IsVisible = false;
            //            }
            //        }
            //        else
            //        {
            //            CommandToolBar[17].IsVisible = false;
            //            CommandToolBar[18].IsVisible = false;
            //            CommandToolBar[19].IsVisible = false;
            //            CommandToolBar[20].IsVisible = false;
            //            CommandToolBar[21].IsVisible = false;
            //        }

            //        ShowProgress = true;
            //    }
            //}
            //else
            //{
            //    if (Settings.Default.TrainingRates == null)
            //        Settings.Default.TrainingRates = new ObservableCollection<DNNTrainingRate> { TrainRate };
                
            //    TrainRates = Settings.Default.TrainingRates;

            //    TrainingSchemeEditor dialog = new TrainingSchemeEditor { Path = StorageDirectory };
            //    dialog.tpvm = this;
            //    dialog.DataContext = this;
            //    dialog.buttonTrain.IsEnabled = false;
            //    dialog.Owner = Application.Current.MainWindow;
            //    dialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;
            //    if (dialog.ShowDialog() ?? false)
            //        Settings.Default.TrainingRates = TrainRates;
            //}

            //Settings.Default.Save();
        }

        private void StrategyButtonClick(object? sender, RoutedEventArgs e)
        {
            if (Settings.Default.TrainingStrategies == null)
            {
                var rate = Settings.Default.TraininingRate != null ? Settings.Default.TraininingRate : new DNNTrainingRate();
                var strategy = new DNNTrainingStrategy(1, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, rate.Momentum, rate.Beta2, rate.Gamma, rate.L2Penalty, rate.Dropout, rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation);
                Settings.Default.TrainingStrategies = new ObservableCollection<DNNTrainingStrategy> { strategy };
                Settings.Default.Save();
            }

            TrainingStrategies = Settings.Default.TrainingStrategies;
            
            //TrainingStrategiesEditor dialog = new TrainingStrategiesEditor { Path = StorageDirectory };
            //dialog.tpvm = this;
            //dialog.DataContext = this;
            //dialog.buttonOk.IsEnabled = true;
            //dialog.Owner = Application.Current.MainWindow;
            //dialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;

            //if (dialog.ShowDialog() ?? false)
            //{
            //    Settings.Default.TrainingStrategies = TrainingStrategies;
            //    Settings.Default.Save();

            //    Model?.ClearTrainingStrategies();
            //    foreach (DNNTrainingStrategy strategy in TrainingStrategies)
            //        Model?.AddTrainingStrategy(strategy);
            //}
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        private async void ForgetButtonClick(object? sender, RoutedEventArgs e)
        {
            var result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to forget all weights?", "Forget Model Weights", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));

            if (result == MessageBoxResult.Yes)                
            {
                Model?.ResetWeights();
                LayersComboBox_SelectionChanged(sender, null);
            }
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        private async void ClearButtonClick(object? sender, RoutedEventArgs e)
        {
            if (TrainingLog?.Count > 0)
            {
                //var sb = new StringBuilder();
                //foreach (DNNTrainingResult row in TrainingLog)
                //    sb.AppendLine(row.Epoch.ToString() + "\t" + row.Rate.ToString() + "\t" + row.Dropout.ToString() + row.Cutout.ToString() + "\t" + row.Distortion.ToString() + "\t" + row.HorizontalFlip.ToString() + "\t" + row.VerticalFlip.ToString() + "\t" + row.TrainErrors.ToString() + "\t" + row.TestErrors.ToString() + "\t" + row.AvgTrainLoss.ToString() + "\t" + row.AvgTestLoss.ToString() + "\t" + row.TrainErrors.ToString() + "\t" + row.TestErrors.ToString() + "\t" + row.TestAccuracy.ToString() + "\t" + row.ElapsedTime.ToString());
                //var clipboard = App.MainWindow?.Clipboard;
                //var dataObject = new DataObject();
                //dataObject.Set(DataFormats.Text, sb.ToString());
                //await clipboard!.SetDataObjectAsync(dataObject);

                var result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to clear the log?", "Clear Log", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
                if (result == MessageBoxResult.Yes)
                {
                    TrainingLog.Clear();
                    Model?.ClearLog();
                    RefreshTrainingPlot();
                }
            }
        }

        private async void OpenLayerWeightsButtonClick(object? sender, RoutedEventArgs e)
        {
#if Linux
            if (Model != null && App.MainWindow != null)
            {
                var folder = Path.Combine(DefinitionsDirectory, Model.Name);

                var dialog = new OpenFileDialog
                {
                    AllowMultiple = false,
                    Title = "Load layer weights",
                    Directory = folder,
                    Filters = new List<FileDialogFilter> { new FileDialogFilter() { Name = "Weights|*.bin", Extensions = new List<string> { "bin" } } }
                };

                var stop = false;
                while (!stop)
                {
                    stop = true;

                    var files = await dialog.ShowAsync(App.MainWindow);

                    if (files != null && files.Length > 0)
                    {
                        var path = files[0];

                        if (path.Contains(".bin") && layersComboBox != null)
                        {
                            if (Model?.LoadLayerWeights(path, (uint)layersComboBox.SelectedIndex) == 0)
                            {
                                Dispatcher.UIThread.Post(() => LayersComboBox_SelectionChanged(sender, null), DispatcherPriority.Render);
                                await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are loaded", "Information", MessageBoxButtons.OK));
                                stop = true;
                            }
                            else
                            {
                                stop = false;
                                await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are incompatible", "Choose a different file", MessageBoxButtons.OK));
                            }
                        }
                    }
                }
            }
#else
            var provider = App.MainWindow?.StorageProvider;

            if (Model != null && provider != null && provider.CanOpen)
            {
                var folder = Path.Combine(DefinitionsDirectory, Model.Name);

                var typeWeights = new FilePickerFileType("Weights")
                {
                    Patterns = ["*.bin"]
                };

                var filterList = new List<FilePickerFileType>();
                filterList?.Add(typeWeights);

                var stop = false;
                while (!stop)
                {
                    stop = true;

                    var files = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
                    {
                        AllowMultiple = false,
                        Title = "Load layer weights",
                        SuggestedStartLocation = provider.TryGetFolderFromPathAsync(folder)?.Result,
                        FileTypeFilter = filterList
                    });

                    if (files != null)
                    {
                        var file = files?.SingleOrDefault();
                        var path = file?.TryGetLocalPath();

                        if (path != null)
                        {
                            if (path.EndsWith(".bin"))
                            {
                                if (layersComboBox != null && Model?.LoadLayerWeights(path, (uint)layersComboBox.SelectedIndex) == 0)
                                {
                                    Dispatcher.UIThread.Post(() => LayersComboBox_SelectionChanged(sender, null), DispatcherPriority.Render);
                                    await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are loaded", "Information", MessageBoxButtons.OK));
                                }
                                else
                                {
                                    stop = false;
                                    await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are incompatible", "Choose a different file", MessageBoxButtons.OK));
                                }
                            }
                        }
                    }
                }
            }
#endif
        }

        private async void SaveLayerWeightsButtonClick(object? sender, RoutedEventArgs e)
        {
            if (Model != null && Model.Layers != null && App.MainWindow != null && layersComboBox != null)
            {
                var layerIndex = layersComboBox.SelectedIndex;
#if Linux
                var folder = Path.Combine(DefinitionsDirectory, Model.Name);
                var dialog = new SaveFileDialog
                {
                    InitialFileName = Model.Layers[layerIndex].Name + ".bin",
                    Title = "Save layer weights",
                    Directory = folder,
                    Filters = new List<FileDialogFilter> { new FileDialogFilter() { Name = "Weights|*.bin", Extensions = new List<string> { "bin" } } },
                    DefaultExtension = ".bin",
                    ShowOverwritePrompt = true
                };

                var path = await dialog.ShowAsync(App.MainWindow);
                if (path != null)
                {
                    if (path.EndsWith(".bin"))
                    {
                        if (Model.SaveLayerWeights(path, (ulong)layerIndex) == 0)
                            await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are saved", "Information", MessageBoxButtons.OK));
                        else
                            await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights not saved!", "Information", MessageBoxButtons.OK));
                    }
                }
#else
                var provider = App.MainWindow?.StorageProvider;

                if (provider != null && provider.CanSave)
                {
                    var folder = Path.Combine(DefinitionsDirectory, Model.Name);

                    var typeWeights = new FilePickerFileType("Weights")
                    {
                        Patterns = ["*.bin"]
                    };

                    var filterList = new List<FilePickerFileType>();
                    filterList?.Add(typeWeights);

                    var files = await provider.SaveFilePickerAsync(new FilePickerSaveOptions
                    {
                        SuggestedFileName = Model.Layers[layerIndex].Name,
                        DefaultExtension = "*.bin",
                        Title = "Save layer weights",
                        ShowOverwritePrompt = true,
                        SuggestedStartLocation = provider.TryGetFolderFromPathAsync(folder)?.Result,
                        FileTypeChoices = filterList
                    });

                    if (files != null)
                    {
                        var path = files?.TryGetLocalPath();

                        if (path != null)
                        {
                            if (path.EndsWith(".bin"))
                            {
                                if (Model.SaveLayerWeights(path, (ulong)layerIndex) == 0)
                                    await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights are saved", "Information", MessageBoxButtons.OK));
                                else
                                    await Dispatcher.UIThread.Invoke(() => MessageBox.Show("Layer weights not saved!", "Information", MessageBoxButtons.OK));
                            }
                        }
                    }
                }
#endif
            }
        }

        private async void ForgetLayerWeightsButtonClick(object? sender, RoutedEventArgs e)
        {
            var result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to forget layer weights?", "Forget Layer Weights", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
            if (result == MessageBoxResult.Yes && layersComboBox != null)
            {
                uint index = (uint)layersComboBox.SelectedIndex;
                Model?.ResetLayerWeights((uint)layersComboBox.SelectedIndex);
                LayersComboBox_SelectionChanged(sender, null);
            }
        }

        public void RefreshButtonClick(object? sender, RoutedEventArgs? e)
        {
            LayersComboBox_SelectionChanged(sender, null);
            RefreshTrainingPlot();
        }

        public void LayersComboBox_SelectionChanged(object? sender, SelectionChangedEventArgs? e)
        {
            Dispatcher.UIThread.Invoke(() =>
            {
                if (Model != null && Model.Layers != null && layersComboBox != null && layersComboBox?.SelectedIndex >= 0)
                {
                    var index = layersComboBox.SelectedIndex;
                    if (index < (int)Model.LayerCount)
                    {
                        Settings.Default.SelectedLayer = index;
                        Settings.Default.Save();
                        Model.SelectedIndex = index;

                        ShowSample = Model.TaskState == DNNTaskStates.Running;
                        ShowWeights = Model.Layers[index].WeightCount > 0 || Settings.Default.Timings;
                        ShowWeightsSnapshot = (Model.Layers[index].IsNormLayer && Model.Layers[index].Scaling) || Model.Layers[index].LayerType == DNNLayerTypes.DepthwiseConvolution || Model.Layers[index].LayerType == DNNLayerTypes.ConvolutionTranspose || Model.Layers[index].LayerType == DNNLayerTypes.Convolution || Model.Layers[index].LayerType == DNNLayerTypes.Dense || (Model.Layers[index].LayerType == DNNLayerTypes.Activation && Model.Layers[index].WeightCount > 0);

                        if (index == 0)
                            Model.UpdateLayerInfo((UInt)index, ShowSample);
                        else
                            Model.UpdateLayerInfo((UInt)index, ShowWeightsSnapshot);

                        if (ShowSample)
                        {
                            if (index != 0)
                                Model.UpdateLayerInfo(0ul, ShowSample);

                            InputSnapshot = Model.InputSnapshot;
                            Label = Model.Label;
                        }

                        CommandToolBar[17].IsVisible = !Settings.Default.DisableLocking;
                        CommandToolBar[18].IsVisible = !Settings.Default.DisableLocking;
                        CommandToolBar[19].IsVisible = Model.Layers[index].Lockable && Model.TaskState == DNNTaskStates.Stopped;
                        CommandToolBar[20].IsVisible = Model.Layers[index].Lockable;
                        CommandToolBar[21].IsVisible = Model.Layers[index].Lockable && Model.TaskState == DNNTaskStates.Stopped;

                        sb.Length = 0;
                        sb.Append(stringLayer + Model.Layers[index].Description + nwl);
                        if (Settings.Default.Timings)
                        {
                            if (Model.State == DNNStates.Training)
                            {
                                sb.Append(stringTimings);
                                sb.AppendFormat(stringFprop + nwl, (int)Model.Layers[index].FPropLayerTime, (int)Model.fpropTime);
                                sb.AppendFormat(stringBprop, (int)Model.Layers[index].BPropLayerTime, (int)Model.bpropTime);
                               
                                if (ShowWeightsSnapshot)
                                    sb.AppendFormat(stringUpdate, (int)Model.Layers[index].UpdateLayerTime, (int)Model.updateTime);
                            }
                            else if (Model.State == DNNStates.Testing)
                            {
                                sb.Append(stringTimings);
                                sb.AppendFormat(stringFprop, (int)Model.Layers[index].FPropLayerTime, (int)Model.fpropTime);
                            }
                        }
                        LayerInfo = sb.ToString();

                        sb.Length = 0;
                        sb.Append(stringNeurons);
                        if (Model.Layers[index].NeuronsStats.StdDev >= 0.0f)
                            sb.AppendFormat(stringStdDevPositive, Model.Layers[index].NeuronsStats.StdDev);
                        else
                            sb.AppendFormat(stringStdDevNegative, Model.Layers[index].NeuronsStats.StdDev);
                       
                        if (Model.Layers[index].NeuronsStats.Mean >= 0.0f)
                            sb.AppendFormat(stringMeanPositive, Model.Layers[index].NeuronsStats.Mean);
                        else
                            sb.AppendFormat(stringMeanNegative, Model.Layers[index].NeuronsStats.Mean);
                       
                        if (Model.Layers[index].NeuronsStats.Min >= 0.0f)
                            sb.AppendFormat(stringMminPositive, Model.Layers[index].NeuronsStats.Min);
                        else
                            sb.AppendFormat(stringMinNegative, Model.Layers[index].NeuronsStats.Min);
                      
                        if (Model.Layers[index].NeuronsStats.Max >= 0.0f)
                            sb.AppendFormat(stringMaxPositive, Model.Layers[index].NeuronsStats.Max);
                        else
                            sb.AppendFormat(stringMaxNegative, Model.Layers[index].NeuronsStats.Max);
                                                
                        if (ShowWeightsSnapshot)
                        {
                            WeightsSnapshotX = Model.Layers[index].WeightsSnapshotX;
                            WeightsSnapshotY = Model.Layers[index].WeightsSnapshotY;
                            WeightsSnapshot = Model.Layers[index].WeightsSnapshot;

                            sb.Append(stringWeights);
                           
                            if (Model.Layers[index].WeightsStats.StdDev >= 0.0f)
                                sb.AppendFormat(stringStdDevPositive, Model.Layers[index].WeightsStats.StdDev);
                            else
                                sb.AppendFormat(stringStdDevNegative, Model.Layers[index].WeightsStats.StdDev);
                           
                            if (Model.Layers[index].WeightsStats.Mean >= 0.0f)
                                sb.AppendFormat(stringMeanPositive, Model.Layers[index].WeightsStats.Mean);
                            else
                                sb.AppendFormat(stringMeanNegative, Model.Layers[index].WeightsStats.Mean);
                           
                            if (Model.Layers[index].WeightsStats.Min >= 0.0f)
                                sb.AppendFormat(stringMminPositive, Model.Layers[index].WeightsStats.Min);
                            else
                                sb.AppendFormat(stringMinNegative, Model.Layers[index].WeightsStats.Min);
                          
                            if (Model.Layers[index].WeightsStats.Max >= 0.0f)
                                sb.AppendFormat(stringMaxPositive, Model.Layers[index].WeightsStats.Max);
                            else
                                sb.AppendFormat(stringMaxNegative, Model.Layers[index].WeightsStats.Max);
                          
                            if (Model.Layers[index].HasBias)
                            {
                                sb.Append(stringBiases);
                            
                                if (Model.Layers[index].BiasesStats.StdDev >= 0.0f)
                                    sb.AppendFormat(stringStdDevPositive, Model.Layers[index].BiasesStats.StdDev);
                                else
                                    sb.AppendFormat(stringStdDevNegative, Model.Layers[index].BiasesStats.StdDev);
                               
                                if (Model.Layers[index].BiasesStats.Mean >= 0.0f)
                                    sb.AppendFormat(stringMeanPositive, Model.Layers[index].BiasesStats.Mean);
                                else
                                    sb.AppendFormat(stringMeanNegative, Model.Layers[index].BiasesStats.Mean);
                                
                                if (Model.Layers[index].BiasesStats.Min >= 0.0f)
                                    sb.AppendFormat(stringMminPositive, Model.Layers[index].BiasesStats.Min);
                                else
                                    sb.AppendFormat(stringMinNegative, Model.Layers[index].BiasesStats.Min);
                                
                                if (Model.Layers[index].BiasesStats.Max >= 0.0f)
                                    sb.AppendFormat(stringMaxPositive, Model.Layers[index].BiasesStats.Max);
                                else
                                    sb.AppendFormat(stringMaxNegative, Model.Layers[index].BiasesStats.Max);
                            }
                        }
                        WeightsMinMax = sb.ToString();
                                               
                        if (e != null)
                            e.Handled = true;
                    }
                }
            }, DispatcherPriority.Render);
        }
    }
}