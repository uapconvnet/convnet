using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using Avalonia.Threading;
using System;
using System.IO;

namespace Convnet.Common
{
    public class Attached : AvaloniaObject
    {
        public static readonly AttachedProperty<string> FormattedTextProperty = AvaloniaProperty.RegisterAttached<Attached, AvaloniaObject, string>(nameof(FormattedText), defaultValue: string.Empty, false, Avalonia.Data.BindingMode.TwoWay);

        static Attached()
        {
            FormattedTextProperty.Changed.AddClassHandler<AvaloniaObject>(FormattedTextPropertyChanged);
        }

        //public static readonly DependencyProperty FormattedTextProperty = DependencyProperty.RegisterAttached("FormattedText", typeof(string), typeof(Attached), new FrameworkPropertyMetadata(string.Empty, FrameworkPropertyMetadataOptions.AffectsMeasure, FormattedTextPropertyChanged));


        /// <summary>
        /// Accessor for Attached property <see cref="FormattedTextProperty"/>.
        /// </summary>
        public static void SetFormattedText(AvaloniaObject textBlock, string value)
        {
            textBlock.SetValue(FormattedTextProperty, value);
        }

        /// <summary>
        /// Accessor for Attached property <see cref="FormattedTextProperty"/>.
        /// </summary>
        public static string GetFormattedText(AvaloniaObject textBlock)
        {
            if (textBlock != null)
                return (string)textBlock.GetValue(FormattedTextProperty);
            else
                throw new ArgumentNullException("FormattedText");
        }

        private static void FormattedTextPropertyChanged(AvaloniaObject d, AvaloniaPropertyChangedEventArgs e)
        {
            if (d is TextBlock textBlock)
            {
                var text = (string?)e.NewValue ?? string.Empty;

                using (TextReader sr = new StringReader(string.Format("<Span xml:space=\"preserve\" xmlns=\"https://github.com/avaloniaui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2006/xaml\">{0}</Span>", text)))
                {
                    if (Dispatcher.UIThread.CheckAccess()) //Check if we are already on the UI thread
                    {
                        if (Avalonia.Markup.Xaml.AvaloniaRuntimeXamlLoader.Load(sr.ReadToEnd()) is Span result)
                        {
                            textBlock.Text = string.Empty;
                            textBlock.Inlines = new InlineCollection();
                            textBlock.Inlines?.Add(result);
                            textBlock.UpdateLayout();
                        }
                    }
                    else
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            if (Avalonia.Markup.Xaml.AvaloniaRuntimeXamlLoader.Load(sr.ReadToEnd()) is Span result)
                            {
                                textBlock.Text = string.Empty;
                                textBlock.Inlines = new InlineCollection();
                                textBlock.Inlines?.Add(result);
                                textBlock.UpdateLayout();
                            }
                        }, DispatcherPriority.Render);
                    }
                }
            }
        }
    }
}
