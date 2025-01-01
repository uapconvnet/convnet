using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Interactivity;
using Avalonia.Threading;
using Convnet.Common;
using Convnet.Dialogs;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;
using ReactiveUI;
using System;
using System.Data;
using System.Linq;
using System.Text;
using System.Timers;

using Float = System.Single;
using UInt = System.UInt64;

namespace Convnet.PageViewModels
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
    public class TestPageViewModel : PageViewModelBase, IDisposable
    {
        private static readonly string nwl = Environment.NewLine;

        private string progressText;
        private bool showProgress;
        private string label;
        private bool showSample;
        private DataTable? confusionDataTable;
        private Avalonia.Media.Imaging.WriteableBitmap inputSnapshot;
        private readonly StringBuilder sb;
        private ComboBox dataProviderComboBox;
        private ComboBox costLayersComboBox;
        public Timer RefreshTimer;
        public event EventHandler Open;
        private int flag = 0;

        
        public TestPageViewModel(Interop.DNNModel model) : base(model)
        {
            AddCommandButtons();

            sb = new StringBuilder();

            showProgress = false;
            showSample = false;
            if (Model != null)
                Model.TestProgress += TestProgress;
            Modelhanged += TestPageViewModel_ModelChanged;

            //Dispatcher.UIThread.Post(() => LayerIndexChanged(this, null), DispatcherPriority.Render);
        }

        private void AddCommandButtons()
        {
            Button startButton = new Button
            {
                Name = "ButtonStart",
                Content = ApplicationHelper.LoadFromResource("Play.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(startButton, "Start Testing");
            startButton.Click += StartButtonClick;

            Button stopButton = new Button
            {
                Name = "ButtonStop",
                Content = ApplicationHelper.LoadFromResource("Stop.png"),
                ClickMode = ClickMode.Release,
                IsVisible = false
            };
            ToolTip.SetTip(stopButton, "Stop Testing");
            stopButton.Click += StopButtonClick;
           
            Button pauseButton = new Button
            {
                Name = "ButtonPause",
                Content = ApplicationHelper.LoadFromResource("Pause.png"),
                ClickMode = ClickMode.Release,
                IsVisible = false
            };
            ToolTip.SetTip(pauseButton, "Pause Testing");
            pauseButton.Click += PauseButtonClick;
          
            dataProviderComboBox = new ComboBox
            {
                Name = "ComboBoxDataSet",
                ItemsSource = Enum.GetValues(typeof(DNNDatasets)).Cast<Enum>().ToList(),
                SelectedIndex = (int)Dataset,
                IsEnabled = false
            };
            ToolTip.SetTip(dataProviderComboBox, "Dataset");

            costLayersComboBox = new ComboBox
            {
                Name = "ComboBoxCostLayers"
            };
            costLayersComboBox.Items.Clear();
            for (uint layer = 0u; layer < Model?.CostLayerCount; layer++)
            {
                ComboBoxItem item = new ComboBoxItem
                {
                    Name = "CostLayer" + layer.ToString(),
                    Content = Model.CostLayers[layer].Name,
                    Tag = layer
                };
                costLayersComboBox.Items.Add(item);
            }
            ToolTip.SetTip(costLayersComboBox, "Cost Layer");
            if (Model != null)
            {
                costLayersComboBox.SelectedIndex = (int)Model.CostIndex;
                costLayersComboBox.IsEnabled = Model.CostLayerCount > 1;
            }
            costLayersComboBox.SelectionChanged += CostLayersComboBox_SelectionChanged;
            
            CommandToolBar.Add(startButton);
            CommandToolBar.Add(stopButton);
            CommandToolBar.Add(pauseButton);
            CommandToolBar.Add(new Separator());
            CommandToolBar.Add(dataProviderComboBox);
            CommandToolBar.Add(costLayersComboBox);
        }

        public void CostLayersComboBox_SelectionChanged(object? sender, SelectionChangedEventArgs? e)
        {
            if (costLayersComboBox.SelectedIndex >= 0)
            {
                var costIndex = (uint)costLayersComboBox.SelectedIndex;
                Model?.SetCostIndex(costIndex);
                if (Model?.TaskState != DNNTaskStates.Running && ConfusionDataTable != null)
                {
                    Model?.GetConfusionMatrix();
                    ConfusionDataTable = GetConfusionDataTable();
                    Model?.UpdateCostInfo(costIndex);
                    ProgressText = string.Format("Loss:\t\t{0:N7}" + nwl + "Errors:\t{1:G}" + nwl + "Error:\t\t{2:N2} %" + nwl + "Accuracy:\t{3:N2} %", Model?.CostLayers[costIndex].AvgTestLoss, Model?.CostLayers[costIndex].TestErrors, Model?.CostLayers[costIndex].TestErrorPercentage, (Float)100 - Model?.CostLayers[costIndex].TestErrorPercentage);
                }
            }
        }

        private void TestPageViewModel_ModelChanged(object? sender, EventArgs e)
        {
            if (Model != null)
            {
                Model.TestProgress += TestProgress;
                ShowProgress = false;
                ShowSample = false;
                ConfusionDataTable = null;

                costLayersComboBox.Items.Clear();
                for (uint layer = 0u; layer < Model.CostLayerCount; layer++)
                {
                    ComboBoxItem item = new ComboBoxItem
                    {
                        Name = "CostLayer" + layer.ToString(),
                        Content = Model.CostLayers[layer].Name,
                        Tag = layer
                    };
                    costLayersComboBox.Items.Add(item);
                }
                costLayersComboBox.SelectedIndex = (int)Model.CostIndex;
                costLayersComboBox.IsEnabled = Model.CostLayerCount > 1;

                dataProviderComboBox.SelectedIndex = (int)Dataset;
            }
            //Dispatcher.UIThread.Post(() => LayerIndexChanged(this, null), DispatcherPriority.Render);
        }

        private void TestProgress(UInt BatchSize, UInt SampleIndex, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors, DNNStates State, DNNTaskStates TaskState)
        {
            if (flag == 0 && State != DNNStates.Completed)
            {
                Dispatcher.UIThread.Post(() =>
                {
                    ProgressText = string.Format("Sample:\t\t{0:G}" + nwl + "Loss:\t\t{1:N7}" + nwl + "Errors:\t\t{2:G}" + nwl + "Error:\t\t{3:N2} %" + nwl + "Accuracy:\t{4:N2} %", SampleIndex, AvgTestLoss, TestErrors, TestErrorPercentage, TestAccuracy);

                    if (Model != null)
                    {
                        Model.UpdateLayerInfo(0ul, true);
                        InputSnapshot = Model.InputSnapshot;
                        Label = Model.Label;
                    }
                });
            }
            else
            {
                Dispatcher.UIThread.Post(() =>
                {
                    ProgressText = string.Format("Loss:\t\t{0:N7}" + nwl + "Errors:\t\t{1:G}" + nwl + "Error:\t\t{2:N2} %" + nwl + "Accuracy:\t{3:N2} %", AvgTestLoss, TestErrors, TestErrorPercentage, TestAccuracy);

                    flag = 1;
                    RefreshTimer.Stop();
                    RefreshTimer.Elapsed -= new ElapsedEventHandler(RefreshTimer_Elapsed);
                    RefreshTimer.Dispose();

                    Model?.Stop();
                    Model?.SetCostIndex((uint)costLayersComboBox.SelectedIndex);
                    Model?.GetConfusionMatrix();
                    ConfusionDataTable = GetConfusionDataTable();

                    ToolTip.SetTip(CommandToolBar[0], "Start Testing");

                    CommandToolBar[0].IsVisible = true;
                    CommandToolBar[1].IsVisible = false;
                    CommandToolBar[2].IsVisible = false;

                    IsValid = true;
                    ShowSample = false;
                });
            }
        }

        public DNNTrainingRate TestRate
        {
            get => Settings.Default.TestRate ?? new DNNTrainingRate(DNNOptimizers.NAG, 0.9f, 0.0005f, 0, 0.999f, 0.000001f, 128, 1, 32, 32, 0, 4, 4, 1, 200, 1, 0.05f, 0.0001f, 0.1f, 0.003f, 1, 1, false, false, 0, 0, false, 0, 0, 0, 0, DNNInterpolations.Cubic, 10, 12);
            private set
            {
                if (value == Settings.Default.TestRate)
                    return;

                Settings.Default.TestRate = value;
                this.RaisePropertyChanged(nameof(TestRate));
            }
        }

        public DataTable? ConfusionDataTable
        {
            get => confusionDataTable;
            set => this.RaiseAndSetIfChanged(ref confusionDataTable, value);
        }

        private DataTable? GetConfusionDataTable()
        {
            DataTable? table = null;

            if (Model?.ConfusionMatrix != null)
            {
                table = new DataTable("ConfusionTable");
                uint classCount = (uint)Model.ClassCount;
                uint labelIndex = (uint)Model.LabelIndex;

                table.BeginInit();
                table.Columns.Add("RowHeader", typeof(string));
                for (uint c = 0; c < classCount; c++)
                {
                    table.Columns.Add(Model.LabelsCollection[labelIndex][c], typeof(uint));
                }
                table.EndInit();

                table.BeginLoadData();
                for (uint r = 0; r < classCount; r++)
                {
                    DataRow row = table.NewRow();
                    row.BeginEdit();
                    object[] rowCollection = new object[classCount + 1];
                    rowCollection[0] = Model.LabelsCollection[labelIndex][r].ToString().Replace("_", "__");
                    for (uint c = 0; c < classCount; c++)
                    {
                        rowCollection[c + 1] = Model.ConfusionMatrix[r*classCount+c];
                    }

                    row.ItemArray = rowCollection;
                    row.EndEdit();
                    table.Rows.Add(row);
                }
                table.EndLoadData();
            }

            return table;
        }

        public bool ShowProgress
        {
            get => showProgress;
            set => this.RaiseAndSetIfChanged(ref showProgress, value);
        }

        public bool ShowSample
        {
            get => showSample;
            set => this.RaiseAndSetIfChanged(ref showSample, value);
        }

        public Avalonia.Media.Imaging.WriteableBitmap InputSnapshot
        {
            get => inputSnapshot;
            set => this.RaiseAndSetIfChanged(ref inputSnapshot, value);
        }

        public string Label
        {
            get => label;
            set => this.RaiseAndSetIfChanged(ref label, value);
        }

        public string ProgressText
        {
            get => progressText;
            set => this.RaiseAndSetIfChanged(ref progressText, value);
        }

        public override string DisplayName => "Test";

        public override void Reset()
        {
            ProgressText = string.Empty;
            Label = string.Empty;
        }

        private void RefreshTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            
        }

        private void StartButtonClick(object? sender, RoutedEventArgs e)
        {
            Dispatcher.UIThread.Post(async () =>
            {
                if (Model?.TaskState == DNNTaskStates.Running)
                {

                    await MessageBox.Show("You must stop training first.", "Information", MessageBoxButtons.OK);

                    return;
                }

                if (Model?.TaskState == DNNTaskStates.Stopped)
                {
                    TestParameters dialog = new TestParameters
                    {
                        Model = this.Model,
                        Path = DefinitionsDirectory,
                        IsEnabled = true,
                        Rate = TestRate,
                        tpvm = this,
                    };

                    await dialog.ShowDialog(App.MainWindow);

                    if (dialog.DialogResult)
                    {
                        IsValid = false;

                        TestRate = dialog.Rate;
                        Settings.Default.Save();

                        flag = 0;
                        Model.AddTrainingRate(new DNNTrainingRate(dialog.Rate.Optimizer, dialog.Rate.Momentum, dialog.Rate.Beta2, dialog.Rate.L2Penalty, dialog.Rate.Dropout, dialog.Rate.Eps, dialog.Rate.N, dialog.Rate.D, dialog.Rate.H, dialog.Rate.W, dialog.Rate.PadD, dialog.Rate.PadH, dialog.Rate.PadW, 1, 1, dialog.Rate.EpochMultiplier, dialog.Rate.MaximumRate, dialog.Rate.MinimumRate, dialog.Rate.FinalRate, dialog.Rate.Gamma, dialog.Rate.DecayAfterEpochs, dialog.Rate.DecayFactor, dialog.Rate.HorizontalFlip, dialog.Rate.VerticalFlip, dialog.Rate.InputDropout, dialog.Rate.Cutout, dialog.Rate.CutMix, dialog.Rate.AutoAugment, dialog.Rate.ColorCast, dialog.Rate.ColorAngle, dialog.Rate.Distortion, dialog.Rate.Interpolation, dialog.Rate.Scaling, dialog.Rate.Rotation), true, 1, Model.TrainingSamples);
                        Model.SetCostIndex((uint)costLayersComboBox.SelectedIndex);
                        Model.Start(false);
                        RefreshTimer = new Timer(1000.0);
                        RefreshTimer.Elapsed += RefreshTimer_Elapsed;

                        CommandToolBar[0].IsVisible = false;
                        CommandToolBar[1].IsVisible = true;
                        CommandToolBar[2].IsVisible = true;

                        ShowProgress = true;
                        ShowSample = true;
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
                    }
                }
            });
        }

        private async void StopButtonClick(object? sender, RoutedEventArgs e)
        {
            if (Model?.TaskState != DNNTaskStates.Stopped)
            {
                var stop = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to stop?", "Stop Testing", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
                if (stop == MessageBoxResult.Yes)
                {
                    RefreshTimer.Stop();
                    RefreshTimer.Elapsed -= new ElapsedEventHandler(RefreshTimer_Elapsed);
                    RefreshTimer.Dispose();

                    Model?.Stop();
                    ConfusionDataTable = null;

                    ToolTip.SetTip(CommandToolBar[0], "Start Testing");
                    CommandToolBar[0].IsVisible = true;
                    CommandToolBar[1].IsVisible = false;
                    CommandToolBar[2].IsVisible = false;

                    IsValid = true;
                    ShowProgress = false;
                    ShowSample = false;
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

                    ToolTip.SetTip(CommandToolBar[0], "Resume Testing");
                    CommandToolBar[0].IsVisible = true;
                    CommandToolBar[1].IsVisible = true;
                    CommandToolBar[2].IsVisible = false;
                }
            });
        }

        private void OpenButtonClick(object? sender, RoutedEventArgs e)
        {
            Open?.Invoke(this, EventArgs.Empty);
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                    RefreshTimer.Dispose();
                    confusionDataTable?.Dispose();
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~TestPageViewModel()
        // {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
