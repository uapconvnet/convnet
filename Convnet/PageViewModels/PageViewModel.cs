using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Convnet.Common;
using Convnet.Properties;
using CsvHelper;
using CustomMessageBox.Avalonia;
using Interop;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using Float = System.Single;
using UInt = System.UInt64;

namespace Convnet.PageViewModels
{
    public enum ViewModels
    {
        Edit = 0,
        Test = 1,
        Train = 2
    }

    public class PageViewModel : PageViewModelBase
    {
        public event EventHandler? PageChange;
        
        public void DocumentationCommand()
        {
            ApplicationHelper.OpenBrowser("https://github.com/uapconvnet/convnet.git");
        }

        public void Cut()
        {
           
            //if (PageVM != null && PageVM.Pages != null)
            //{
            //    //var epvm = MainView.PageViews.Items[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;

            //    var epvm = PageVM.Pages[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;
            //    if (epvm != null)
            //    {
            //        var topLevel = TopLevel.GetTopLevel(this);
            //        if (FocusManager != null)
            //        {
            //            var elem = FocusManager.GetFocusedElement();

            //        }
            //    }
            //}
        }

        public bool CanCut()
        {
            //if (PageVM != null && PageVM.Pages != null)
            //{
            //    //var epvm = MainView.PageViews.Items[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;

            //    var epvm = PageVM.Pages[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;
            //    if (epvm != null)
            //    {
            //        var topLevel = TopLevel.GetTopLevel(this);
            //        if (FocusManager != null)
            //        {
            //            var elem = FocusManager.GetFocusedElement();
            //            return true;
            //        }
            //    }
            //}

            return true;
        }

        public PageViewModel(DNNModel model) : base(model)
        {
            Settings.Default.PropertyChanged += Default_PropertyChanged;

            progressBarMinimum = 0.0;
            progressBarMaximum = 100.0;

            if (Model != null)
            {
                Model.TrainProgress += TrainProgress;
                Model.TestProgress += TestProgress;

                var EditPageVM = new EditPageViewModel(Model);
                EditPageVM.Open += PageVM_Open;
                EditPageVM.Save += PageVM_SaveAs;
                EditPageVM.Modelhanged += EditPageVM_ModelChanged;

                var TestPageVM = new TestPageViewModel(Model);
                TestPageVM.Open += PageVM_Open;

                var TrainPageVM = new TrainPageViewModel(Model);
                TrainPageVM.Open += PageVM_Open;
                TrainPageVM.Save += PageVM_Save;

                Pages = new ReadOnlyCollection<PageViewModelBase?>([EditPageVM, TestPageVM, TrainPageVM]);
                CurrentPage = Pages[Settings.Default.CurrentPage];
            }
        }

        private void Default_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            Settings.Default.Save();
        }

        private async void PageVM_Open(object? sender, EventArgs e)
        {
            var path = string.Empty;

            if (Model != null)
            {
                var folder = DefinitionsDirectory;
                if (Directory.Exists(Path.Combine(DefinitionsDirectory, Model.Name)))
                    folder = Path.Combine(DefinitionsDirectory, Model.Name);

#if Linux
                var dialog = new Avalonia.Controls.OpenFileDialog
                {
                    AllowMultiple = false,
                    Title = "Load",
                    Directory = folder
                };

                if (CurrentPage is TrainPageViewModel)
                {
                    dialog.Filters.Add(new Avalonia.Controls.FileDialogFilter() { Name = "Weights|*.bin", Extensions = new List<string> { "bin" } });
                    dialog.Filters.Add(new Avalonia.Controls.FileDialogFilter() { Name = "Log|*.csv", Extensions = new List<string> { "csv" } });
                }
                if (CurrentPage is EditPageViewModel)
                {
                    dialog.Filters.Add(new Avalonia.Controls.FileDialogFilter() { Name = "Definition|*.txt", Extensions = new List<string> { "txt" } });
                    dialog.Filters.Add(new Avalonia.Controls.FileDialogFilter() { Name = "C#|*.cs", Extensions = new List<string> { "cs" } });
                }

                var files = await dialog.ShowAsync(App.MainWindow);

                if (files != null && files.Length > 0)
                    path = files[0];
#else
                var provider = App.MainWindow?.StorageProvider;

                if (provider != null && provider.CanOpen)
                {
                    var typeWeights = new FilePickerFileType("Weights")
                    {
                        Patterns = ["*.bin"]
                    };
                    var typeLog = new FilePickerFileType("Log")
                    {
                        Patterns = ["*.csv"]
                    };

                    var typeDefinition = new FilePickerFileType("Definition")
                    {
                        Patterns = ["*.txt"]
                    };
                    var typeCSharp = new FilePickerFileType("C#")
                    {
                        Patterns = ["*.cs"]
                    };

                    var filterList = new List<FilePickerFileType>();
                    if (CurrentPage is TrainPageViewModel)
                        filterList?.AddRange([typeWeights, typeLog]);
                    if (CurrentPage is EditPageViewModel)
                        filterList?.AddRange([typeDefinition, typeCSharp]);

                    var files = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
                    {
                        AllowMultiple = false,
                        Title = "Load",
                        SuggestedStartLocation = provider.TryGetFolderFromPathAsync(folder)?.Result,
                        FileTypeFilter = filterList
                    }); 

                    var file = files?.SingleOrDefault();

                    path = file?.TryGetLocalPath();
                }
#endif
                if (path != null)
                {
                    if (path.EndsWith(".csv"))
                    {
                        if (CurrentPage is TrainPageViewModel tpvm)
                        {
                            var backup = Settings.Default.TrainingLog != null ? new ObservableCollection<DNNTrainingResult>(Settings.Default.TrainingLog) : new ObservableCollection<DNNTrainingResult>();

                            try
                            {
                                var config = new CsvHelper.Configuration.CsvConfiguration(CultureInfo.CurrentCulture)
                                {
                                    HasHeaderRecord = true,
                                    DetectDelimiter = true,
                                    DetectDelimiterValues = [";"],
                                    Delimiter = ";"
                                };

                                using (var reader = new StreamReader(path, true))
                                using (var csv = new CsvReader(reader, config))
                                {
                                    var records = csv.GetRecords<DNNTrainingResult>();

                                    if (Settings.Default.TrainingLog?.Count > 0)
                                    {
                                        var result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to clear the log?", "Clear Log", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));

                                        if (result == MessageBoxResult.Yes)
                                        {
                                            Settings.Default.TrainingLog.Clear();
                                            Model?.ClearLog();
                                        }
                                    }

                                    foreach (var record in records)
                                        Settings.Default.TrainingLog?.Add(record);
                                }

                                Model?.LoadLog(path);
                            }
                            catch (Exception ex)
                            {
                                Settings.Default.TrainingLog = backup;
                                Dispatcher.UIThread.Post(() => MessageBox.Show(ex.Message, "Information", MessageBoxButtons.OK));
                            }

                            Settings.Default.Save();

                            tpvm.RefreshTrainingPlot();
                            Dispatcher.UIThread.Post(() => MessageBox.Show(path + " is loaded", "Information", MessageBoxButtons.OK));
                        }
                    }
                    else if (path.EndsWith(".bin"))
                    {
                        if (CurrentPage is TrainPageViewModel tpvm)
                        {
                            if (tpvm.Model != null)
                            {
                                if (tpvm.Model.LoadWeights(path, Settings.Default.PersistOptimizer) == 0)
                                {
                                    Dispatcher.UIThread.Post(() =>
                                    {
                                        tpvm.Optimizer = tpvm.Model.Optimizer;
                                        tpvm.RefreshButtonClick(this, null);
                                        MessageBox.Show(path + " is loaded", "Information", MessageBoxButtons.OK);
                                    });
                                }
                                else
                                    Dispatcher.UIThread.Post(() => MessageBox.Show(path + " is incompatible", "Information", MessageBoxButtons.OK));
                            }
                        }
                    }
                    else if (path.EndsWith(".txt"))
                    {
                        if (CurrentPage is EditPageViewModel epvm)
                        {
                            if (epvm.Model != null)
                            {
                                var reader = new StreamReader(path, true);
                                var definition = reader.ReadToEnd().Trim();
                                epvm.Definition = definition;
                                Settings.Default.DefinitionEditing = definition;
                                Settings.Default.Save();
                                Dispatcher.UIThread.Post(() => MessageBox.Show(path + " is loaded", "Information", MessageBoxButtons.OK));
                            }
                        }
                    }
                    else if (path.EndsWith(".cs"))
                    {
                        if (CurrentPage is EditPageViewModel epvm)
                        {
                            if (epvm.Model != null)
                            {
                                var reader = new StreamReader(path, true);
                                var script = reader.ReadToEnd().Trim();
                                epvm.Script = script;
                                Settings.Default.Script = script;
                                Settings.Default.Save();
                                Dispatcher.UIThread.Post(() => MessageBox.Show(path + " is loaded", "Information", MessageBoxButtons.OK));
                            }
                        }
                    }
                }
            }
        }

        private async void PageVM_Save(object? sender, EventArgs e)
        {
            if (Model != null)
            {
                var path = Path.Combine(DefinitionsDirectory, Model.Name);
                var shortFileName = @"(" + Model.Dataset.ToString().ToLower() + @")" + (Settings.Default.PersistOptimizer ? @"(" + Model.Optimizer.ToString().ToLower() + @").bin" : @".bin");
                var fileName = Model.Name + @"-" + shortFileName;
                               
                MessageBoxResult result = MessageBoxResult.Yes;
                if (File.Exists(Path.Combine(StateDirectory, fileName)))
                    result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you want to overwrite the existing file?", "File already exists", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
               
                if (result == MessageBoxResult.Yes)
                    if (Model.SaveWeights(Path.Combine(StateDirectory, fileName), Settings.Default.PersistOptimizer) == 0)
                    {
                        File.Copy(Path.Combine(StateDirectory, fileName), Path.Combine(path, shortFileName), true);
                        Dispatcher.UIThread.Post(() => MessageBox.Show("Weights are saved", "Information", MessageBoxButtons.OK));
                    }
            }
        }

        private void PageVM_SaveAs(object? sender, EventArgs e)
        {
        }

        private void TrainProgress(DNNOptimizers Optimizer, UInt BatchSize, UInt Cycle, UInt TotalCycles, UInt Epoch, UInt TotalEpochs, bool HorizontalMirror, bool VerticalMirror, Float InputDropOut, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorRadius, Float Distortion, DNNInterpolations Interpolation, Float Scaling, Float Rotation, UInt SampleIndex, Float Rate, Float Momentum, Float Beta2, Float Gamma, Float L2Penalty, Float DropOut, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float Accuracy, UInt64 TestErrors, DNNStates NetworkState, DNNTaskStates TaskState)
        {
            if (Model != null)
            {
                switch (NetworkState)
                {
                    case DNNStates.Training:
                        ProgressValue = (double)(SampleIndex + 1ul) / Model.AdjustedTrainSamplesCount;
                        ProgressBarMaximum = Model.AdjustedTrainSamplesCount;
                        break;

                    case DNNStates.Testing:
                        ProgressValue = (double)(SampleIndex + 1ul) / Model.AdjustedTestSamplesCount;
                        ProgressBarMaximum = Model.AdjustedTestSamplesCount;
                        break;

                    default:
                        ProgressBarValue = 0.0;
                        SampleIndex = 0ul;
                        break;
                }

                ProgressBarValue = SampleIndex;
                Duration = Model.DurationString;
                SampleRate = Model.SampleRate.ToString("N1");
            }
        }

        private void TestProgress(UInt BatchSize, UInt SampleIndex, Float AvgTestLoss, Float TestErrorPercentage, Float Accuracy, UInt TestErrors, DNNStates NetworkState, DNNTaskStates TaskState)
        {
            if (Model != null)
            {
                if (NetworkState == DNNStates.Testing)
                {
                    ProgressValue = (double)(SampleIndex + 1ul) / Model.AdjustedTestSamplesCount;
                    ProgressBarMaximum = Model.AdjustedTestSamplesCount;
                }
                else
                {
                    ProgressValue = 0.0;
                    SampleIndex = 0ul;
                }

                ProgressBarValue = SampleIndex;
                Duration = Model.DurationString;
                SampleRate = Model.SampleRate.ToString("N1");
            }
        }

        private void EditPageVM_ModelChanged(object? sender, EventArgs e)
        {
            if (Pages != null && Pages.Count > (int)ViewModels.Train)
            {
                Model = Pages[(int)ViewModels.Edit]?.Model;
                if (Model != null)
                {
                    Model.TrainProgress += TrainProgress;
                    Model.TestProgress += TestProgress;
                    Pages[(int)ViewModels.Train].Model = Model;
                    Pages[(int)ViewModels.Test].Model = Model;
                }
            }
        }

        public ViewModels? CurrentModel
        {
            get => (Pages != null && CurrentPage != null) ? (ViewModels?)Pages.IndexOf(CurrentPage) : null;  
        }

        public override string DisplayName => "Main";
                
        private string? sampleRate;
        public string? SampleRate
        {
            get => sampleRate;
            set => this.RaiseAndSetIfChanged(ref sampleRate, value);
        }

        private string? duration;
        public string? Duration
        {
            get => duration;
            set => this.RaiseAndSetIfChanged(ref duration, value);
        }

        private double progressBarMinimum = 0;
        public double ProgressBarMinimum
        {
            get => progressBarMinimum;
            set => this.RaiseAndSetIfChanged(ref progressBarMinimum, value);
        }

        private double progressBarMaximum = 100;
        public double ProgressBarMaximum
        {
            get => progressBarMaximum;
            set => this.RaiseAndSetIfChanged(ref progressBarMaximum, value);
        }

        private double progressBarValue = 0;
        public double ProgressBarValue
        {
            get => progressBarValue;
            set => this.RaiseAndSetIfChanged(ref progressBarValue, value);
        }

        private double progressValue = 0;
        public double ProgressValue
        {
            get => progressValue;
            set => this.RaiseAndSetIfChanged(ref progressValue, value);
        }

        public ReadOnlyCollection<PageViewModelBase?>? Pages { get; }

        private PageViewModelBase? currentPage;
        public PageViewModelBase? CurrentPage
        {
            get => currentPage; 
            set
            {
                if (value == currentPage || value == null)
                    return;

                this.RaiseAndSetIfChanged(ref currentPage, value);
                if (currentPage != null && Pages != null)
                {
                    CommandToolBar = currentPage.CommandToolBar;
                    CommandToolBarVisibility = currentPage.CommandToolBarVisibility;
                    Settings.Default.CurrentPage = Pages.IndexOf(currentPage);
                    Settings.Default.Save();

                    OnPageChange();
                }
            }
        }

        public void OnPageChange()
        {
            PageChange?.Invoke(this, EventArgs.Empty);

            if (Settings.Default.CurrentPage == (int)ViewModels.Edit && Pages != null)
            {
                var vm = Pages[(int)ViewModels.Edit] as EditPageViewModel;
                vm?.CheckButtonClick(this, new Avalonia.Interactivity.RoutedEventArgs());
            }

            if (Settings.Default.CurrentPage == (int)ViewModels.Test)
            {
                var testPVM = Pages?[(int)ViewModels.Test] as TestPageViewModel;

                if (testPVM?.Model != null)
                {
                    testPVM.CommandToolBar[0].IsVisible = testPVM.Model.TaskState == DNNTaskStates.Stopped;
                    testPVM.CommandToolBar[1].IsVisible = false;
                    testPVM.CommandToolBar[2].IsVisible = false;
                }

                if (CostLayers?.Count > 1)
                    testPVM?.CostLayersComboBox_SelectionChanged(this, null);
            }

            if (Settings.Default.CurrentPage == (int)ViewModels.Train && CostLayers?.Count > 1)
                (Pages?[(int)ViewModels.Train] as TrainPageViewModel)?.CostLayersComboBox_SelectionChanged(this, null);
        }

        public override void Reset()
        {
            if (Pages != null)
                foreach (PageViewModelBase? page in Pages)
                    page?.Reset();
        }
    }
}