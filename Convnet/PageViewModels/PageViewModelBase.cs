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

        private ObservableCollection<Control> commandToolBar = new ObservableCollection<Control>();
        private bool commandToolBarVisibility = false;
        private bool isValid = true;

        public static string? ApplicationPath { get; } = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

        public static string StorageDirectory { get; } = Path.Combine(Environment.GetFolderPath(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Environment.SpecialFolder.MyDocuments : Environment.SpecialFolder.UserProfile), "convnet");

        public static string StateDirectory { get; } = Path.Combine(StorageDirectory, "state");

        public static string DefinitionsDirectory { get; } = Path.Combine(StorageDirectory, "definitions");

        public static string ScriptsDirectory { get; } = Path.Combine(StorageDirectory, "scripts");

        public static string ScriptPath { get; } = Path.Combine(ScriptsDirectory, "bin", Mode, Framework);

        public static IEnumerable<DNNOptimizers> GetOptimizers => Enum.GetValues<DNNOptimizers>().Cast<DNNOptimizers>();

        public static IEnumerable<DNNInterpolations> GetInterpolations => Enum.GetValues<DNNInterpolations>().Cast<DNNInterpolations>();

        public abstract string DisplayName { get; }
       
        public ObservableCollection<Control> CommandToolBar
        {
            get => commandToolBar;
            set => this.RaiseAndSetIfChanged(ref commandToolBar, value);
        }

        public bool CommandToolBarVisibility
        {
            get => commandToolBarVisibility;
            set => this.RaiseAndSetIfChanged(ref commandToolBarVisibility, value);
        }
        
        public bool IsValid
        {
            get => isValid;
            set => this.RaiseAndSetIfChanged(ref isValid, value);
        }

        private void CommandToolBarCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            CommandToolBarVisibility = CommandToolBar.Count > 0;
        }

        public abstract void Reset();

        protected PageViewModelBase()
        {
            CommandToolBar.CollectionChanged += new NotifyCollectionChangedEventHandler(CommandToolBarCollectionChanged);
        }
    }
}
