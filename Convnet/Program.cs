﻿using Avalonia;
using Avalonia.Dialogs;
using Avalonia.ReactiveUI;
using System;

namespace Convnet
{
    internal sealed class Program
    {
        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args) => BuildAvaloniaApp()
            .StartWithClassicDesktopLifetime(args);

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
        {
            return AppBuilder.Configure<App>()
                        .UsePlatformDetect()
//.With(new X11PlatformOptions { UseDBusFilePicker = false }) // to disable FreeDesktop file picker
//#if Linux
//                        .UseManagedSystemDialogs()
//#endif
                        .WithInterFont()
                        .LogToTrace()
                        .UseReactiveUI();
        }
    }
}
