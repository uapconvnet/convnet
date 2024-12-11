using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using AvaloniaEdit;
using Convnet.Common;
using Convnet.PageViewModels;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;
using ReactiveUI;
using System;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Convnet.PageViews
{
    public partial class MainWindow : Window
    {
        const string Framework = "net9.0";
#if DEBUG
        const string Mode = "Debug";
#else
        const string Mode = "Release";
#endif
        public static string? ApplicationPath { get; } = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        public static string StorageDirectory { get; } = Path.Combine(Environment.GetFolderPath(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Environment.SpecialFolder.MyDocuments :Environment.SpecialFolder.UserProfile), "convnet");
        public static string StateDirectory { get; } = Path.Combine(StorageDirectory, "state");
        public static string DefinitionsDirectory { get; } = Path.Combine(StorageDirectory, "definitions");
        public static string ScriptsDirectory { get; } = Path.Combine(StorageDirectory, "scripts");
        
        public PageViewModel? PageVM;

        public ReactiveCommand<Unit, Unit> CutCommand { get; }
        public void Cut()
        {
            //var elem = TopLevel.GetTopLevel(this).GetFocusedElement();
            ApplicationCommands.Cut.Execute(null, null);
        }
        public bool CanCut
        {
            get => ApplicationCommands.Cut.CanExecute(null, null);
        }

        

        private bool PersistLog(string path)
        {
            try
            {
                const string delim = ";";
                var sb = new StringBuilder();
                sb.AppendLine(
                        "Cycle" + delim +
                        "Epoch" + delim +
                        "GroupIndex" + delim +
                        "CostIndex" + delim +
                        "CostName" + delim +
                        "N" + delim +
                        "D" + delim +
                        "H" + delim +
                        "W" + delim +
                        "PadD" + delim +
                        "PadH" + delim +
                        "PadW" + delim +
                        "Optimizer" + delim +
                        "Rate" + delim +
                        "Eps" + delim +
                        "Momentum" + delim +
                        "Beta2" + delim +
                        "Gamma" + delim +
                        "L2Penalty" + delim +
                        "Dropout" + delim +
                        "InputDropout" + delim +
                        "Cutout" + delim +
                        "CutMix" + delim +
                        "AutoAugment" + delim +
                        "HorizontalFlip" + delim +
                        "VerticalFlip" + delim +
                        "ColorCast" + delim +
                        "ColorAngle" + delim +
                        "Distortion" + delim +
                        "Interpolation" + delim +
                        "Scaling" + delim +
                        "Rotation" + delim +
                        "AvgTrainLoss" + delim +
                        "TrainErrors" + delim +
                        "TrainErrorPercentage" + delim +
                        "TrainAccuracy" + delim +
                        "AvgTestLoss" + delim +
                        "TestErrors" + delim +
                        "TestErrorPercentage" + delim +
                        "TestAccuracy" + delim +
                        "ElapsedMilliSeconds" + delim +
                        "ElapsedTime");

                if (Settings.Default.TrainingLog != null)
                foreach (var row in Settings.Default.TrainingLog)
                    sb.AppendLine(
                        row.Cycle.ToString() + delim +
                        row.Epoch.ToString() + delim +
                        row.GroupIndex.ToString() + delim +
                        row.CostIndex.ToString() + delim +
                        row.CostName.ToString() + delim +
                        row.N.ToString() + delim +
                        row.D.ToString() + delim +
                        row.H.ToString() + delim +
                        row.W.ToString() + delim +
                        row.PadD.ToString() + delim +
                        row.PadH.ToString() + delim +
                        row.PadW.ToString() + delim +
                        row.Optimizer.ToString() + delim +
                        row.Rate.ToString() + delim +
                        row.Eps.ToString() + delim +
                        row.Momentum.ToString() + delim +
                        row.Beta2.ToString() + delim +
                        row.Gamma.ToString() + delim +
                        row.L2Penalty.ToString() + delim +
                        row.Dropout.ToString() + delim +
                        row.InputDropout.ToString() + delim +
                        row.Cutout.ToString() + delim +
                        row.CutMix.ToString() + delim +
                        row.AutoAugment.ToString() + delim +
                        row.HorizontalFlip.ToString() + delim +
                        row.VerticalFlip.ToString() + delim +
                        row.ColorCast.ToString() + delim +
                        row.ColorAngle.ToString() + delim +
                        row.Distortion.ToString() + delim +
                        row.Interpolation.ToString() + delim +
                        row.Scaling.ToString() + delim +
                        row.Rotation.ToString() + delim +
                        row.AvgTrainLoss.ToString() + delim +
                        row.TrainErrors.ToString() + delim +
                        row.TrainErrorPercentage.ToString() + delim +
                        row.TrainAccuracy.ToString() + delim +
                        row.AvgTestLoss.ToString() + delim +
                        row.TestErrors.ToString() + delim +
                        row.TestErrorPercentage.ToString() + delim +
                        row.TestAccuracy.ToString() + delim +
                        row.ElapsedMilliSeconds.ToString() + delim +
                        row.ElapsedTime.ToString());

                File.WriteAllText(path, sb.ToString());
                
                if (PageVM != null && PageVM.Model != null)
                    PageVM.Model.LoadLog(path);

                return true;
            }
            catch (Exception ex)
            {
                //Mouse.OverrideCursor = null;
                MessageBox.Show(ex.ToString(), "Exception occured", MessageBoxButtons.OK);
            }

            return false;
        }


        public MainWindow()
        {
            InitializeComponent();

            CutCommand = ReactiveCommand.Create(Cut, this.WhenAnyValue(x => x.CanCut));
            //var cmdKey = GetPlatformCommandKey();

            //var file = new MenuItem();
            //file.Header = "_File";

            //var edit = new MenuItem();
            //edit.Header = "_Edit";

            //var cut = new MenuItem { Header = "Cut", InputGesture = new KeyGesture(Key.X, cmdKey) };
            //var copy = new MenuItem { Header = "Copy", InputGesture = new KeyGesture(Key.C, cmdKey) };
            //var paste = new MenuItem { Header = "Paste", InputGesture = new KeyGesture(Key.V, cmdKey) };
            //var delete = new MenuItem { Header = "Delete", InputGesture = new KeyGesture(Key.Delete) };
            //var selectall = new MenuItem { Header = "Select All", InputGesture = new KeyGesture(Key.A, cmdKey) };
            //var undo = new MenuItem { Header = "Undo", InputGesture = new KeyGesture(Key.Z, cmdKey) };
            //var redo = new MenuItem { Header = "Redo", InputGesture = new KeyGesture(Key.Y, cmdKey) };

            //cut.Icon = ImageHelper.LoadFromResource("Cut.png");
            //paste.Icon = ImageHelper.LoadFromResource("Paste.png");
            //copy.Icon = ImageHelper.LoadFromResource("Copy.png");
            //delete.Icon = ImageHelper.LoadFromResource("Cancel.png");
            //selectall.Icon = ImageHelper.LoadFromResource("SelectAll.png");
            //undo.Icon = ImageHelper.LoadFromResource("Undo.png");
            //redo.Icon = ImageHelper.LoadFromResource("Redo.png");

            ////cut.Command = ApplicationCommands.Cut;
            ////paste.Command = ApplicationCommands.Paste;
            ////copy.Command = ApplicationCommands.Copy;
            ////delete.Command = ApplicationCommands.Delete;
            ////selectall.Command = ApplicationCommands.SelectAll;
            ////undo.Command = ApplicationCommands.Undo;
            ////redo.Command = ApplicationCommands.Redo;

            ////cut.Click += (s, e) => { if (CanCut) Dispatcher.UIThread.Post(() => Cut()); };
            ////paste.Click += (s, e) => { if (CanPaste) Dispatcher.UIThread.Post(() => Paste()); };
            ////copy.Click += (s, e) => { if (CanCopy) Dispatcher.UIThread.Post(() => Copy()); };
            ////delete.Click += (s, e) => { if (CanDelete) Dispatcher.UIThread.Post(() => Delete()); };
            ////selectall.Click += (s, e) => { if (CanSelectAll) Dispatcher.UIThread.Post(() => SelectAll()); };
            ////undo.Click += (s, e) => { if (CanUndo) Dispatcher.UIThread.Post(() => Undo()); };
            ////redo.Click += (s, e) => { if (CanRedo) Dispatcher.UIThread.Post(() => Redo()); };

            //edit.Items.Add(cut);
            //edit.Items.Add(copy);
            //edit.Items.Add(paste);
            //edit.Items.Add(delete);
            //edit.Items.Add(new Separator());
            //edit.Items.Add(selectall);
            //edit.Items.Add(new Separator());
            //edit.Items.Add(undo);
            //edit.Items.Add(redo);

            //menuMain.Items.Add(file);
            //menuMain.Items.Add(edit);



            Directory.CreateDirectory(StorageDirectory);
            Directory.CreateDirectory(DefinitionsDirectory);
            Directory.CreateDirectory(StateDirectory);

            if (!Directory.Exists(ScriptsDirectory) && ApplicationPath != null)
            {
                Directory.CreateDirectory(ScriptsDirectory);
                ApplicationHelper.CopyDir(Path.Combine(ApplicationPath.Replace(Path.Combine("Convnet", "bin", (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "x64" : ""), Mode, Framework), ""), "Scripts"), ScriptsDirectory);
            }

            var fileName = Path.Combine(StateDirectory, Settings.Default.ModelNameActive + ".txt");
            var backupModelName = "resnet-3-2-6-channelzeropad-relu";

            if (!File.Exists(Path.Combine(StateDirectory, backupModelName + ".txt")) && ApplicationPath != null)
                File.Copy(Path.Combine(ApplicationPath, "Resources", "state", backupModelName + ".txt"), Path.Combine(StateDirectory, backupModelName + ".txt"), true);

            if (ApplicationPath != null)
            { 
                if (!File.Exists(fileName) || !File.ReadLines(Path.Combine(StateDirectory, backupModelName + ".txt")).SequenceEqual(File.ReadLines(Path.Combine(ApplicationPath, "Resources", "state", backupModelName + ".txt"))))
                {
                    Directory.CreateDirectory(Path.Combine(DefinitionsDirectory, backupModelName));
                    File.Copy(Path.Combine(ApplicationPath, "Resources", "state", backupModelName + ".txt"), Path.Combine(DefinitionsDirectory, backupModelName + ".txt"), true);

                    fileName = Path.Combine(StateDirectory, backupModelName + ".txt");
                    Settings.Default.ModelNameActive = backupModelName;
                    Settings.Default.DefinitionActive = File.ReadAllText(fileName);
                    Settings.Default.Optimizer = (int)DNNOptimizers.NAG;
                    Settings.Default.Save();
                }
            }

            try
            {
                var model = new DNNModel(Settings.Default.DefinitionActive);

                if (model != null)
                {
                    PageVM = new PageViewModel(model);

                    if (PageVM != null && PageVM.Model != null)
                    {
                        model.BackgroundColor = Settings.Default.BackgroundColor;
                        model.BlockSize = (ulong)Settings.Default.PixelSize;
                        model.TrainingStrategies = Settings.Default.TrainingStrategies;
                        model.ClearTrainingStrategies();
                        if (Settings.Default.TrainingStrategies != null)
                            foreach (DNNTrainingStrategy strategy in Settings.Default.TrainingStrategies)
                                model.AddTrainingStrategy(strategy);
                        model.SetFormat(Settings.Default.PlainFormat);
                        model.SetOptimizer((DNNOptimizers)Settings.Default.Optimizer);
                        model.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                        model.SetUseTrainingStrategy(Settings.Default.UseTrainingStrategy);
                        model.SetDisableLocking(Settings.Default.DisableLocking);
                        model.SetShuffleCount((ulong)Math.Round(Settings.Default.Shuffle));

                        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".csv");
                        if (PersistLog(path))
                            File.Delete(path);

                        var dataset = PageVM.Model.Dataset.ToString().ToLower();
                        var optimizer = PageVM.Model.Optimizer.ToString().ToLower();

                        var fileNamePersistOptimizer = Path.Combine(StateDirectory, Settings.Default.ModelNameActive + "-(" + dataset + ")(" + optimizer + @").bin");
                        var fileNameNoOptimizer = Path.Combine(StateDirectory, Settings.Default.ModelNameActive + "-(" + dataset + ").bin");

                        var fileNameOptimizer = Settings.Default.PersistOptimizer ? fileNamePersistOptimizer : fileNameNoOptimizer;
                        var fileNameOptimizerInverse = Settings.Default.PersistOptimizer ? fileNameNoOptimizer : fileNamePersistOptimizer;

                        if (PageVM.Model.LoadWeights(fileNameOptimizer, Settings.Default.PersistOptimizer) != 0)
                            if (PageVM.Model.LoadWeights(fileNameOptimizerInverse, !Settings.Default.PersistOptimizer) == 0)
                            {
                                Settings.Default.PersistOptimizer = !Settings.Default.PersistOptimizer;
                                Settings.Default.Save();
                                PageVM.Model.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                            }

                        Title = PageVM.Model.Name + " - Convnet Explorer";
                        
                        DataContext = PageVM;

                        //switch ((int)Math.Round(Settings.Default.PrioritySetter))
                        //{
                        //    case 1:
                        //        PrioritySlider.ToolTip = "Low";
                        //        break;
                        //    case 2:
                        //        PrioritySlider.ToolTip = "Below Normal";
                        //        break;
                        //    case 3:
                        //        PrioritySlider.ToolTip = "Normal";
                        //        break;
                        //    case 4:
                        //        PrioritySlider.ToolTip = "Above Normal";
                        //        break;
                        //    case 5:
                        //        PrioritySlider.ToolTip = "High";
                        //        break;
                        //    case 6:
                        //        PrioritySlider.ToolTip = "Realtime";
                        //        break;
                        //}
                    }
                    else
                        MessageBox.Show("Failed to create the PageViewModel: " + Settings.Default.ModelNameActive, "Error", MessageBoxButtons.OK);
                }
                else
                {
                    // try backup model
                    if (ApplicationPath != null)
                        File.Copy(Path.Combine(ApplicationPath, "Resources", "state", backupModelName + ".txt"), Path.Combine(StateDirectory, backupModelName + ".txt"), true);
                    fileName = Path.Combine(StateDirectory, backupModelName + ".txt");
                    Settings.Default.ModelNameActive = backupModelName;
                    Settings.Default.DefinitionActive = File.ReadAllText(fileName);
                    Settings.Default.Optimizer = (int)DNNOptimizers.NAG;
                    Settings.Default.Save();

                    model = new DNNModel(Settings.Default.DefinitionActive);

                    if (model == null)
                    {
                        MessageBox.Show("Failed to create the Model: " + Settings.Default.ModelNameActive, "Error", MessageBoxButtons.OK);
                        return;
                    }

                    PageVM = new PageViewModel(model);

                    if (PageVM != null && PageVM.Model != null)
                    {
                        model.BackgroundColor = Settings.Default.BackgroundColor;
                        model.BlockSize = (ulong)Settings.Default.PixelSize;
                        model.TrainingStrategies = Settings.Default.TrainingStrategies;
                        model.ClearTrainingStrategies();
                        if (Settings.Default.TrainingStrategies != null)
                            foreach (DNNTrainingStrategy strategy in Settings.Default.TrainingStrategies)
                                model.AddTrainingStrategy(strategy);
                        model.SetFormat(Settings.Default.PlainFormat);
                        model.SetOptimizer((DNNOptimizers)Settings.Default.Optimizer);
                        model.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                        model.SetUseTrainingStrategy(Settings.Default.UseTrainingStrategy);
                        model.SetDisableLocking(Settings.Default.DisableLocking);
                        model.SetShuffleCount((ulong)Math.Round(Settings.Default.Shuffle));

                        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".csv");
                        if (PersistLog(path))
                            File.Delete(path);

                        var dataset = PageVM.Model.Dataset.ToString().ToLower();
                        var optimizer = PageVM.Model.Optimizer.ToString().ToLower();

                        var fileNamePersistOptimizer = Path.Combine(StateDirectory, Settings.Default.ModelNameActive + "-(" + dataset + ")(" + optimizer + @").bin");
                        var fileNameNoOptimizer = Path.Combine(StateDirectory, Settings.Default.ModelNameActive + "-(" + dataset + ").bin");

                        var fileNameOptimizer = Settings.Default.PersistOptimizer ? fileNamePersistOptimizer : fileNameNoOptimizer;
                        var fileNameOptimizerInverse = Settings.Default.PersistOptimizer ? fileNameNoOptimizer : fileNamePersistOptimizer;

                        if (PageVM.Model.LoadWeights(fileNameOptimizer, Settings.Default.PersistOptimizer) != 0)
                            if (PageVM.Model.LoadWeights(fileNameOptimizerInverse, !Settings.Default.PersistOptimizer) == 0)
                            {
                                Settings.Default.PersistOptimizer = !Settings.Default.PersistOptimizer;
                                Settings.Default.Save();
                                PageVM.Model.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                            }

                        Title = PageVM.Model.Name + " - Convnet Explorer";
                        DataContext = PageVM;

                        //switch ((int)Math.Round(Settings.Default.PrioritySetter))
                        //{
                        //    case 1:
                        //        PrioritySlider.ToolTip = "Low";
                        //        break;
                        //    case 2:
                        //        PrioritySlider.ToolTip = "Below Normal";
                        //        break;
                        //    case 3:
                        //        PrioritySlider.ToolTip = "Normal";
                        //        break;
                        //    case 4:
                        //        PrioritySlider.ToolTip = "Above Normal";
                        //        break;
                        //    case 5:
                        //        PrioritySlider.ToolTip = "High";
                        //        break;
                        //    case 6:
                        //        PrioritySlider.ToolTip = "Realtime";
                        //        break;
                        //}
                    }
                    else
                        MessageBox.Show("Failed to create the PageViewModel: " + Settings.Default.ModelNameActive, "Error", MessageBoxButtons.OK);
                }
            }
            catch (Exception exception)
            {
                if (exception != null)
                    if (exception.InnerException != null)
                        MessageBox.Show(exception.Message + Environment.NewLine + Environment.NewLine + exception.GetBaseException().Message + Environment.NewLine + Environment.NewLine + exception.InnerException.Message + Environment.NewLine + Environment.NewLine + "An error occured while loading the Model:" + Settings.Default.ModelNameActive, "Information", MessageBoxButtons.OK);
                    else
                        MessageBox.Show(exception.Message + Environment.NewLine + Environment.NewLine + exception.GetBaseException().Message + Environment.NewLine + Environment.NewLine + "An error occured while loading the Model:" + Settings.Default.ModelNameActive, "Information", MessageBoxButtons.OK);
            }
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        public void MainWindow_Closing(object? sender, Avalonia.Controls.WindowClosingEventArgs e)
        {
            if (App.ShowCloseApplicationDialog)
            {
                MessageBoxResult exit = MessageBoxResult.Yes;
                using (CancellationTokenSource source = new CancellationTokenSource())
                {
                    if (Dispatcher.UIThread.CheckAccess()) //Check if we are already on the UI thread
                    {
                        var dialog = new MessageBox("Do you really want to exit?", "Exit application");

                        dialog.Show<MessageBoxResult>(new MessageBoxButton<MessageBoxResult>("Yes", MessageBoxResult.Yes, SpecialButtonRole.None), new MessageBoxButton<MessageBoxResult>("No", MessageBoxResult.No, SpecialButtonRole.IsCancel)).ContinueWith(t =>
                        {
                            exit = t.Result;
                            source.Cancel();
                        });

                        Dispatcher.UIThread.MainLoop(source.Token);
                    }
                    else
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            var dialog = new MessageBox("Do you really want to exit?", "Exit application");
                            dialog.Show<MessageBoxResult>(new MessageBoxButton<MessageBoxResult>("Yes", MessageBoxResult.Yes, SpecialButtonRole.None), new MessageBoxButton<MessageBoxResult>("No", MessageBoxResult.No, SpecialButtonRole.IsCancel)).ContinueWith(t =>
                            {
                                exit = t.Result;
                                source.Cancel();
                            });
                        });

                        while (!source.IsCancellationRequested) { } //Loop until dialog is closed
                    }
                }
                if (exit == MessageBoxResult.Yes)
                {
                    if (PageVM?.Model?.TaskState != DNNTaskStates.Stopped)
                        PageVM?.Model?.Stop();

                    MessageBoxResult save = MessageBoxResult.Yes;
                    using (CancellationTokenSource source = new CancellationTokenSource())
                    {
                        if (Dispatcher.UIThread.CheckAccess()) //Check if we are already on the UI thread
                        {
                            var dialog = new MessageBox("Do you want to save the state?", "Save state");

                            dialog.Show<MessageBoxResult>(new MessageBoxButton<MessageBoxResult>("Yes", MessageBoxResult.Yes, SpecialButtonRole.IsDefault), new MessageBoxButton<MessageBoxResult>("No", MessageBoxResult.No, SpecialButtonRole.IsCancel)).ContinueWith(t =>
                            {
                                save = t.Result;
                                source.Cancel();
                            });

                            Dispatcher.UIThread.MainLoop(source.Token);
                        }
                        else
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                var dialog = new MessageBox("Do you want to save the state?", "Save state");
                                dialog.Show<MessageBoxResult>(new MessageBoxButton<MessageBoxResult>("Yes", MessageBoxResult.Yes, SpecialButtonRole.IsDefault), new MessageBoxButton<MessageBoxResult>("No", MessageBoxResult.No, SpecialButtonRole.IsCancel)).ContinueWith(t =>
                                {
                                    save = t.Result;
                                    source.Cancel();
                                });
                            });

                            while (!source.IsCancellationRequested) { } //Loop until dialog is closed
                        }
                    }
                    if (save == MessageBoxResult.Yes)
                    {
                        var dataset = PageVM?.Model?.Dataset.ToString().ToLower();
                        var optimizer = PageVM?.Model?.Optimizer.ToString().ToLower();
                        var fileName = Path.Combine(StateDirectory, PageVM?.Model?.Name + @"-(" + dataset + @")" + (Settings.Default.PersistOptimizer ? (@"(" + optimizer + @").bin") : @".bin"));

                        PageVM?.Model?.SaveWeights(fileName, Settings.Default.PersistOptimizer);
                    }

                    Settings.Default.Save();
                    e.Cancel = false;
                }
                else
                    e.Cancel = true;
            }
            else
            {
                Settings.Default.Save();
                e.Cancel = false;
            }
        }
    }
}