using Avalonia; // AppBuilder
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using System;
using System.Diagnostics;
using System.Reflection;
using System.Threading;


namespace Convnet
{
    public partial class App : Application
    {
        public static readonly bool SingleInstanceApp = true;
        public static readonly bool ShowCloseApplicationDialog = true;
        
        public static PageViews.MainWindow? MainWindow = null;
        public event EventHandler<ShutdownRequestedEventArgs>? ShutdownRequested;
        private static readonly SingleInstanceMutex sim = new SingleInstanceMutex();

        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }

        public override void OnFrameworkInitializationCompleted()
        {
            //// Get an array of plugins to remove
            //var dataValidationPluginsToRemove = BindingPlugins.DataValidators.OfType<DataAnnotationsValidationPlugin>().ToArray();

            //// remove each entry found
            //foreach (var plugin in dataValidationPluginsToRemove)
            //{
            //    BindingPlugins.DataValidators.Remove(plugin);
            //}

            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                if (SingleInstanceApp && sim.IsOtherInstanceRunning)
                    return;

                desktop.ShutdownRequested += AppShutdownRequested;
                desktop.MainWindow = new Convnet.PageViews.MainWindow
                {
                };

                if (desktop.MainWindow != null)
                {
                    App.MainWindow = desktop.MainWindow as Convnet.PageViews.MainWindow;
                }
            }
            
            base.OnFrameworkInitializationCompleted();
        }

        protected virtual void AppShutdownRequested(object? sender, ShutdownRequestedEventArgs e)
        {
            Debug.WriteLine($"App.{nameof(AppShutdownRequested)}");
            OnShutdownRequested(e);
        }

        protected virtual void OnShutdownRequested(ShutdownRequestedEventArgs e)
        {
            ShutdownRequested?.Invoke(this, e);
        }

       
        /// <summary>
        /// Represents a <see cref="SingleInstanceMutex"/> class.
        /// </summary>
        public partial class SingleInstanceMutex : IDisposable
        {
            #region Fields

            /// <summary>
            /// Indicator whether another instance of this application is running or not.
            /// </summary>
            private readonly bool isNoOtherInstanceRunning;

            /// <summary>
            /// The <see cref="Mutex"/> used to ask for other instances of this application.
            /// </summary>
            private Mutex? singleInstanceMutex = null;

            /// <summary>
            /// An indicator whether this object is beeing actively disposed or not.
            /// </summary>
            private bool disposed;

            #endregion

            #region Constructor

            /// <summary>
            /// Initializes a new instance of the <see cref="SingleInstanceMutex"/> class.
            /// </summary>
            public SingleInstanceMutex()
            {
                singleInstanceMutex = new Mutex(true, Assembly.GetCallingAssembly().FullName, out isNoOtherInstanceRunning);
            }

            #endregion

            #region Properties

            /// <summary>
            /// Gets an indicator whether another instance of the application is running or not.
            /// </summary>
            public bool IsOtherInstanceRunning
            {
                get
                {
                    return !isNoOtherInstanceRunning;
                }
            }

            #endregion

            #region Methods

            /// <summary>
            /// Closes the <see cref="SingleInstanceMutex"/>.
            /// </summary>
            public void Close()
            {
                ThrowIfDisposed();
                singleInstanceMutex?.Close();
            }

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            private void Dispose(bool disposing)
            {
                if (!disposed)
                {
                    /* Release unmanaged ressources */

                    if (disposing)
                    {
                        /* Release managed ressources */
                        Close();
                    }

                    disposed = true;
                }
            }

            /// <summary>
            /// Throws an exception if something is tried to be done with an already disposed object.
            /// </summary>
            /// <remarks>
            /// All public methods of the class must first call this.
            /// </remarks>
            public void ThrowIfDisposed()
            {
                if (disposed)
                {
                    throw new ObjectDisposedException(GetType().Name);
                }
            }

            #endregion
        }
    }
}