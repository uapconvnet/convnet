using Avalonia.Controls;
using Interop;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Convnet.PageViewModels
{
    public abstract class PageViewModelBase : ReactiveObject
    {
        const string Framework = @"net10.0";
#if DEBUG
        const string Mode = "Debug";
#else
        const string Mode = @"Release";
#endif

        public static string? ApplicationPath { get; } = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        public static string StorageDirectory { get; } = Path.Combine(Environment.GetFolderPath(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Environment.SpecialFolder.MyDocuments : Environment.SpecialFolder.UserProfile), "convnet");
        public static string StateDirectory { get; } = Path.Combine(StorageDirectory, "state");
        public static string DefinitionsDirectory { get; } = Path.Combine(StorageDirectory, "definitions");
        public static string ScriptsDirectory { get; } = Path.Combine(StorageDirectory, "scripts");
        public static string ScriptPath { get; } = Path.Combine(ScriptsDirectory, "bin", Mode, Framework);
        public static IEnumerable<DNNOptimizers> GetOptimizers => Enum.GetValues<DNNOptimizers>().Cast<DNNOptimizers>();
        public static IEnumerable<DNNInterpolations> GetInterpolations => Enum.GetValues<DNNInterpolations>().Cast<DNNInterpolations>();

        public abstract string DisplayName { get; }

        public abstract void Reset();

        private ObservableCollection<Control> commandToolBar;
        public ObservableCollection<Control> CommandToolBar
        {
            get => commandToolBar;
            set => this.RaiseAndSetIfChanged(ref commandToolBar, value);
        }

        private bool commandToolBarVisibility;
        public bool CommandToolBarVisibility
        {
            get => commandToolBarVisibility;
            set => this.RaiseAndSetIfChanged(ref commandToolBarVisibility, value);
        }

        private bool isValid = true;
        public bool IsValid
        {
            get => isValid;
            set => this.RaiseAndSetIfChanged(ref isValid, value);
        }

        private void CommandToolBarCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            if (CommandToolBar.Count > 0)
                CommandToolBarVisibility = true;
            else
                CommandToolBarVisibility = false;
        }

        protected PageViewModelBase()
        {

            commandToolBarVisibility = false;
            commandToolBar = new ObservableCollection<Control>();
            commandToolBar.CollectionChanged += new NotifyCollectionChangedEventHandler(CommandToolBarCollectionChanged);
        }
    }
}
