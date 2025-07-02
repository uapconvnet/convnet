using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using AvaloniaEdit.Document;
using Convnet.Common;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;
using ReactiveUI;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;

namespace Convnet.PageViewModels
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
    public class EditPageViewModel : PageViewModelBase
    {
#if DEBUG
        const string Mode = "Debug";
#else
        const string Mode = "Release";
#endif
        public event EventHandler? Open;
        public event EventHandler? Save;

        private string modelName = Settings.Default.ModelNameActive;
        private string definition = Settings.Default.DefinitionEditing;
        private bool definitionStatus = false;
        private bool canSynchronize = false;
        private int selectionStart = 0;
        private int selectionLength = 0;
        private TextLocation textLocationDefinition = new(0, 0);
        private TextLocation textLocationScript = new(0, 0);
        private string filePath = string.Empty;
        private bool wordWrap = false;
        private bool showLineNumbers = true;
        private string script = Settings.Default.Script;
        private bool dirty = true;
        private static bool initAction = true;
        private readonly DispatcherTimer clickWaitTimer;
       
        public EditPageViewModel(DNNModel model) : base(model)
        {
            initAction = true;
            clickWaitTimer = new DispatcherTimer(new TimeSpan(0, 0, 0, 0, 50), DispatcherPriority.Background, MouseWaitTimer_Tick);
           
            AddCommandButtons();
        }
      
        private void AddCommandButtons()
        {
            var openButton = new Button
            {
                Name = "ButtonOpen",
                Content = ApplicationHelper.LoadFromResource("OpenFile.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(openButton, "Open");
            openButton.Click += OpenButtonClick;
            
            var saveButton = new Button
            {
                Name = "ButtonSave",
                Content = ApplicationHelper.LoadFromResource("SaveAs.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(saveButton, "Save");
            saveButton.Click += SaveButtonClick;

            var checkButton = new Button
            {
                Name = "ButtonCheck",
                Content = ApplicationHelper.LoadFromResource("SpellCheck.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(checkButton, "Check");
            checkButton.Click += CheckButtonClick;
            
            var synchronizeButton = new Button
            {
                Name = "ButtonSynchronize",
                Content = ApplicationHelper.LoadFromResource("Sync.png"),
                ClickMode = ClickMode.Release
            };
            ToolTip.SetTip(synchronizeButton, "Synchronize");
            synchronizeButton.Click += SynchronizeButtonClick;
            var binding = new Avalonia.Data.Binding("CanSynchronize")
            {
                Converter = new Converters.BooleanToVisibilityConverter(),
                Source = this
            };
            synchronizeButton.Bind(Button.IsVisibleProperty, binding);

            var scriptsButton = new Button
            {
                Name = "ButtonScripts",
                Content = ApplicationHelper.LoadFromResource("Calculator.png"),
                ClickMode = ClickMode.Release,
            };
            ToolTip.SetTip(scriptsButton, "Run Script");
            scriptsButton.Click += ScriptsButtonClick;
           
            var visualStudioButton = new Button
            {
                Name = "ButtonVisualStudio",
                Content = ApplicationHelper.LoadFromResource("VisualStudio.png"),
                ClickMode = ClickMode.Release,
            };
            ToolTip.SetTip(visualStudioButton, "Open in Visual Studio");
            visualStudioButton.Click += VisualStudioButtonClick;

            CommandToolBar.Add(openButton);
            CommandToolBar.Add(saveButton);
            CommandToolBar.Add(checkButton);
            CommandToolBar.Add(synchronizeButton);
            CommandToolBar.Add(scriptsButton);
            CommandToolBar.Add(visualStudioButton);
        }

        public override string DisplayName => "Edit";

        public override void Reset()
        {
            DefinitionStatus = false;
        }

        public string Definition
        {
            get => definition; 
            set
            {
                if (value.Equals(definition) || value.Trim().Length < 3)
                    return;

                this.RaiseAndSetIfChanged(ref definition, value);

                Settings.Default.DefinitionEditing = definition;
                ModelName = definition.Split(Environment.NewLine.ToCharArray(), StringSplitOptions.RemoveEmptyEntries)[0].Trim().Replace("[", "").Replace("]", "").Trim();
                DefinitionStatus = false;
            }
        }

        public string FilePath
        {
            get => filePath;
            set => this.RaiseAndSetIfChanged(ref filePath, value);
        }

        public bool WordWrap
        {
            get => wordWrap;
            set => this.RaiseAndSetIfChanged(ref wordWrap, value);
        }

        public bool ShowLineNumbers
        {
            get => showLineNumbers;
            set => this.RaiseAndSetIfChanged(ref showLineNumbers, value);
        }

        public int SelectionStart
        {
            get => selectionStart;
            set => this.RaiseAndSetIfChanged(ref selectionStart, value);
        }

        public int SelectionLength
        {
            get => selectionLength;
            set => this.RaiseAndSetIfChanged(ref selectionLength, value);
        }

        public TextLocation TextLocationDefinition
        {
            get => textLocationDefinition;
            set => this.RaiseAndSetIfChanged(ref textLocationDefinition, value);
        }

        public TextLocation TextLocationScript
        {
            get => textLocationScript;
            set => this.RaiseAndSetIfChanged(ref textLocationScript, value);
        }

        public bool DefinitionStatus
        {
            get => definitionStatus;
            set
            {
                if (value != definitionStatus)
                {
                    this.RaiseAndSetIfChanged(ref definitionStatus, value);
                    var sameDefinition = Definition.ToLower().Equals(Settings.Default.DefinitionActive.ToLower());
                    CanSynchronize = definitionStatus && !sameDefinition && Model != null && Model.TaskState == DNNTaskStates.Stopped;
                }
            }
        }

        public bool CanSynchronize
        {
            get => canSynchronize;
            set => this.RaiseAndSetIfChanged(ref canSynchronize, value);
        }

        public string ModelName
        {
            get => modelName;
            set
            {
                if (value.Equals(modelName))
                    return;

                if (value.Trim().All(c => char.IsLetterOrDigit(c) || c == '-' || c == '(' || c == ')'))
                   this.RaiseAndSetIfChanged(ref modelName, value.Trim());
            }
        }
     
        public string Script
        {
            get => script;
            set
            {
                if (value.Equals(script))
                    return;

                this.RaiseAndSetIfChanged(ref script, value);
                Settings.Default.Script = script;
                dirty = true;
            }
        }

        private static readonly string[] separator = [Environment.NewLine];

        private void OpenButtonClick(object? sender, RoutedEventArgs e)
        {
            Open?.Invoke(this, EventArgs.Empty);
        }

        private void SaveButtonClick(object? sender, RoutedEventArgs e)
        {
            Save?.Invoke(this, EventArgs.Empty);
        }

        public void CheckButtonClick(object? sender, RoutedEventArgs e)
        {
            DefinitionStatus = CheckDefinition();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        private async void SynchronizeButtonClick(object? sender, RoutedEventArgs e)
        {
            try
            {
                var modelname = Definition.Split(Environment.NewLine.ToCharArray(), StringSplitOptions.RemoveEmptyEntries)[0].Trim().Replace("[", "").Replace("]", "").Trim();
                var notSameModelName = modelname != Model?.Name || modelname != ModelName;
                var sameDefinition = Definition.ToLower(CultureInfo.CurrentCulture).Equals(Settings.Default.DefinitionActive.ToLower(CultureInfo.CurrentCulture));
                var pathDefinition = Path.Combine(DefinitionsDirectory, modelname + ".txt");
                var pathStateDefinition = Path.Combine(StateDirectory, modelname + ".txt");
                var pathWeightsDirectory = Path.Combine(DefinitionsDirectory, modelname);
                var pathWeights = Settings.Default.PersistOptimizer ? Path.Combine(pathWeightsDirectory, Dataset.ToString().ToLower(CultureInfo.CurrentCulture) + "-" + Settings.Default.Optimizer.ToString().ToLower(CultureInfo.CurrentCulture) + @".bin") : Path.Combine(pathWeightsDirectory, Dataset.ToString().ToLower(CultureInfo.CurrentCulture) + ".bin");

                if (notSameModelName || !sameDefinition)
                {
                    var ok = true;
                    if (File.Exists(Path.Combine(DefinitionsDirectory, modelname + ".txt")))
                    {
                        var overwrite = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("File already exists! Overwrite?", "File already exists", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2), DispatcherPriority.Render);
                        ok = overwrite == MessageBoxResult.Yes;
                    }

                    if (ok)
                    {
                        File.WriteAllText(pathDefinition, Definition);
                        File.WriteAllText(pathStateDefinition, Definition);

                        if (!Directory.Exists(pathWeightsDirectory))
                            Directory.CreateDirectory(pathWeightsDirectory);

                        var reloadWeights = false;
                        if (sameDefinition)
                        {
                            var keepWeights = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Keep model weights?", "Same Model", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button1), DispatcherPriority.Render);
                            if (keepWeights == MessageBoxResult.Yes)
                            {
                                Model?.SaveWeights(pathWeights, Settings.Default.PersistOptimizer);
                                reloadWeights = true;
                            }
                        }

                        try
                        {
                            Model?.Dispose();
                            Model = new DNNModel(Definition)
                            {
                                BackgroundColor = Settings.Default.BackgroundColor,
                                BlockSize = (UInt64)Settings.Default.PixelSize,
                                TrainingStrategies = Settings.Default.TrainingStrategies != null ? Settings.Default.TrainingStrategies : new System.Collections.ObjectModel.ObservableCollection<DNNTrainingStrategy>()
                            };
                            Model?.ClearTrainingStrategies();
                            if (Settings.Default.TrainingStrategies != null)
                                foreach (DNNTrainingStrategy strategy in Settings.Default.TrainingStrategies)
                                    Model?.AddTrainingStrategy(strategy);
                            Model?.SetFormat(Settings.Default.PlainFormat);
                            Model?.SetOptimizer((DNNOptimizers)Settings.Default.Optimizer);
                            Model?.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                            Model?.SetUseTrainingStrategy(Settings.Default.UseTrainingStrategy);
                            Model?.SetDisableLocking(Settings.Default.DisableLocking);
                            Model?.SetShuffleCount((ulong)Math.Round(Settings.Default.Shuffle));

                            if (reloadWeights)
                                Model?.LoadWeights(pathWeights, Settings.Default.PersistOptimizer);

                            ModelName = modelname;
                            Settings.Default.ModelNameActive = Model?.Name;
                            Settings.Default.DefinitionEditing = Definition;
                            Settings.Default.DefinitionActive = Definition;
                            Settings.Default.Script = Script;
                            Settings.Default.ScriptActive = Script;

                            Settings.Default.Save();

                            if (App.MainWindow != null)
                                App.MainWindow.Title = Model?.Name + " - Convnet Explorer";

                            CanSynchronize = false;

                            GC.Collect(GC.MaxGeneration);
                            GC.WaitForFullGCComplete();
                            
                            Dispatcher.UIThread.Post(() => MessageBox.Show("Model synchronized", "Information", MessageBoxButtons.OK));
                        }
                        catch (Exception ex)
                        {
                            Dispatcher.UIThread.Post(() => MessageBox.Show("An error occured during synchronization:\r\n" + ex.ToString(), "Synchronize Debug Information", MessageBoxButtons.OK), DispatcherPriority.Normal);
                        }
                    }
                }
                else
                {
                    if (notSameModelName)
                    {
                        var ok = true;
                        if (File.Exists(Path.Combine(DefinitionsDirectory, modelname + ".txt")))
                        {
                            var overwrite = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("File already exists! Overwrite?", "File already exists", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
                            ok = overwrite == MessageBoxResult.Yes;
                        }

                        if (ok)
                        {
                            File.WriteAllText(pathDefinition, Definition);
                            File.WriteAllText(pathStateDefinition, Definition);

                            if (!Directory.Exists(pathWeightsDirectory))
                                Directory.CreateDirectory(pathWeightsDirectory);

                            var keepWeights = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Keep model weights?", "Same Model", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button1));
                            if (keepWeights == MessageBoxResult.Yes)
                                Model?.SaveWeights(pathWeights, Settings.Default.PersistOptimizer);

                            try
                            {
                                Model?.Dispose();
                                Model = new DNNModel(Definition);
                                Model?.SetFormat(Settings.Default.PlainFormat);
                                Model?.SetOptimizer((DNNOptimizers)Settings.Default.Optimizer);
                                Model?.SetPersistOptimizer(Settings.Default.PersistOptimizer);
                                Model?.SetDisableLocking(Settings.Default.DisableLocking);
                                Model?.SetShuffleCount((ulong)Math.Round(Settings.Default.Shuffle));
                                Settings.Default.Save();
                                if (Model != null)
                                    Model.BlockSize = (UInt64)Settings.Default.PixelSize;

                                if (keepWeights == MessageBoxResult.Yes)
                                    Model?.LoadWeights(pathWeights, Settings.Default.PersistOptimizer);

                                ModelName = modelname;
                                Settings.Default.ModelNameActive = Model?.Name;
                                Settings.Default.DefinitionEditing = Definition;
                                Settings.Default.DefinitionActive = Definition;
                                Settings.Default.Script = Script;
                                Settings.Default.ScriptActive = Script;

                                Settings.Default.Save();

                                if (App.MainWindow != null)
                                    App.MainWindow.Title = Model?.Name + " - Convnet Explorer";

                                CanSynchronize = false;

                                GC.Collect(GC.MaxGeneration);
                                GC.WaitForFullGCComplete();

                                Dispatcher.UIThread.Post(() => MessageBox.Show("Model synchronized", "Information", MessageBoxButtons.OK));
                            }
                            catch (Exception ex)
                            {
                                Dispatcher.UIThread.Post(() => MessageBox.Show("An error occured during synchronization:\r\n" + ex.ToString(), "Synchronize Debug Information", MessageBoxButtons.OK));
                            }
                        }
                    }
                }

                Settings.Default.Dataset = Model?.Dataset.ToString().ToLower(CultureInfo.CurrentCulture);
                Settings.Default.Save();
            }
            catch (Exception ex)
            {
                Dispatcher.UIThread.Post(() => MessageBox.Show("An error occured during synchronization:\r\n" + ex.ToString(), "Synchronize Debug Information", MessageBoxButtons.OK));
            }
        }

        private void MouseWaitTimer_Tick(object? sender, EventArgs e)
        {
            clickWaitTimer.Stop();

            if (!initAction)
                Dispatcher.UIThread.Post(() => ScriptDialog());
        }

        private void ScriptsButtonClick(object? sender, RoutedEventArgs e)
        {
            initAction = false;
            clickWaitTimer.Start();
        }

        private void VisualStudioButtonClick(object? sender, RoutedEventArgs e)
        {
            var vspath = @"C:\Program Files\Microsoft Visual Studio\2022\";
            var version = @"Community";
            const string common = @"\Common7\IDE\";

            if (!Directory.Exists(vspath))
                vspath = @"C:\Program Files (x86)\Microsoft Visual Studio\2019\";
            if (Directory.Exists(vspath + @"Community" + common))
                version = "Community";
            else if (Directory.Exists(vspath + @"Professional" + common))
                version = "Professional";
            else if (Directory.Exists(vspath + @"Enterprise" + common))
                version = "Enterprise";

            if (version.Length > 1)
            {
                try
                {
                    var ProcStartInfo = new ProcessStartInfo(vspath + version + common + "devenv.exe", ScriptsDirectory + Path.DirectorySeparatorChar + "Scripts.csproj")
                    {
                        WorkingDirectory = ScriptsDirectory,
                        Verb = "runas",
                        UseShellExecute = true,
                        CreateNoWindow = true,
                        RedirectStandardError = false,
                        RedirectStandardOutput = false
                    };
                    
                    Process.Start(ProcStartInfo);
                }
                catch (Exception)
                {
                }
            }
        }

        async Task ScriptsDialogAsync()
        {
            await ProcessAsyncHelper.RunAsync(new ProcessStartInfo(Path.Combine(ScriptPath, RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "Scripts" : "Scripts.exe")), null);

            var fileName = Path.Combine(ScriptPath, "script.txt");
            var fileInfo = new FileInfo(fileName);

            if (fileInfo.Exists)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    //#pragma warning disable CA1416 // Validate platform compatibility
                    var security = new FileSecurity(fileInfo.FullName, AccessControlSections.Owner | AccessControlSections.Group | AccessControlSections.Access);
                    //var authorizationRules = security.GetAccessRules(true, true, typeof(NTAccount));
                    var owner = security.GetOwner(typeof(NTAccount));
                    if (owner != null)
                        security.ModifyAccessRule(AccessControlModification.Add, new FileSystemAccessRule(owner, FileSystemRights.Modify, AccessControlType.Allow), out bool modified);
                    //#pragma warning restore CA1416 // Validate platform compatibility
                }

                Definition = File.ReadAllText(fileName);
                ModelName = Definition.Split(separator, StringSplitOptions.RemoveEmptyEntries)[0].Replace("[", "").Replace("]", "");
                DefinitionStatus = Dispatcher.UIThread.Invoke(() => CheckDefinition());

                fileInfo.Delete();
            }
        }

        private void ScriptDialog()
        {
            if (dirty)
            {
                IsValid = false;

                try
                {
                    var csproj = "<Project Sdk=\"Microsoft.NET.Sdk\">\r\n\r\n  <PropertyGroup>\r\n    <OutputType>Exe</OutputType>\r\n    <TargetFramework>net9.0</TargetFramework>\r\n    <ImplicitUsings>enable</ImplicitUsings>\r\n    <Nullable>enable</Nullable>\r\n  </PropertyGroup>\r\n\r\n</Project>";
                    File.WriteAllText(Path.Combine(ScriptsDirectory, "Scripts.csproj"), csproj);
                    File.WriteAllText(Path.Combine(ScriptsDirectory, "Program.cs"), Script);

                    var processInfo = new ProcessStartInfo("dotnet", @"build Scripts.csproj -p:Platform=AnyCPU -p:nugetinteractive=true -c:" + Mode + " -fl -flp:logfile=msbuild.log;verbosity=quiet")
                    {
                        WorkingDirectory = ScriptsDirectory,
                        UseShellExecute = RuntimeInformation.IsOSPlatform(OSPlatform.Windows),
                        CreateNoWindow = true,
                        WindowStyle = ProcessWindowStyle.Hidden,
                        Verb = "runas"
                    };

                    File.Delete(Path.Combine(ScriptsDirectory, "msbuild.log"));

                    using (var process = Process.Start(processInfo))
                    {
                        process?.WaitForExit();
                    }

                    var log = File.ReadAllText(Path.Combine(ScriptsDirectory, "msbuild.log"));

                    
                    IsValid = true;
                    dirty = log.Length > 0;
                    if (dirty)
                        Dispatcher.UIThread.Post(() => MessageBox.Show(log, "Build error", MessageBoxButtons.OK));
                }
                catch (Exception ex)
                {
                    IsValid = true;
                    Dispatcher.UIThread.Post(() => MessageBox.Show(ex.Message, "Build failed", MessageBoxButtons.OK));
                }
            }

            try
            {
                if (!dirty)
                {
                    File.Delete(Path.Combine(ScriptPath, "Scripts.deps.json"));
                    Dispatcher.UIThread.Post(async () => await ScriptsDialogAsync());
                }
            }
            catch (Exception exception)
            {
                IsValid = true;
                Dispatcher.UIThread.Post(() => MessageBox.Show(exception.Message, "Load Assembly", MessageBoxButtons.OK));
            }
        }

        private bool CheckDefinition()
        {
            if (Model != null)
            {
                var definition = new StringBuilder(Definition);
                var msg = Model.Check(ref definition);

                Definition = msg.Definition;

                if (msg.Error)
                {
                    TextLocationDefinition = new TextLocation(1, 1);
                    TextLocationDefinition = new TextLocation((int)msg.Row, (int)msg.Column);
                    Dispatcher.UIThread.Invoke(() => MessageBox.Show(msg.Message, "Check", MessageBoxButtons.OK, icon: MessageBoxIcon.Information));
                }
                
                return !msg.Error;
            }

            return false;
        }
    }
}