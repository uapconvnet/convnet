using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using Avalonia.Threading;

namespace Convnet.Common
{
    public class Attached : AvaloniaObject
    {
        public static readonly AttachedProperty<string> FormattedTextProperty = AvaloniaProperty.RegisterAttached<Attached, AvaloniaObject, string>(nameof(FormattedText), defaultValue: string.Empty, false, Avalonia.Data.BindingMode.OneWay);

        static Attached()
        {
            FormattedTextProperty.Changed.AddClassHandler<AvaloniaObject>(FormattedTextPropertyChanged);
        }

        //public static readonly DependencyProperty FormattedTextProperty = DependencyProperty.RegisterAttached("FormattedText", typeof(string), typeof(Attached), new FrameworkPropertyMetadata(string.Empty, FrameworkPropertyMetadataOptions.AffectsMeasure, FormattedTextPropertyChanged));


        /// <summary>
        /// Accessor for Attached property <see cref="FormattedTextProperty"/>.
        /// </summary>
        public static void SetFormattedText(AvaloniaObject d, string value)
        {
            if (d is TextBlock textBlock)
                textBlock.SetValue(FormattedTextProperty, value);
        }

        /// <summary>
        /// Accessor for Attached property <see cref="FormattedTextProperty"/>.
        /// </summary>
        public static string GetFormattedText(AvaloniaObject d)
        {
            if (d is TextBlock textBlock)
                return (string)textBlock.GetValue(FormattedTextProperty);
            
            return string.Empty;
        }

        private static void FormattedTextPropertyChanged(AvaloniaObject d, AvaloniaPropertyChangedEventArgs e)
        {
            Dispatcher.UIThread.Post(() =>
            {
                if (d is TextBlock textBlock)
                {
                    var text = (string?)e.NewValue ?? string.Empty;

                    if (Avalonia.Markup.Xaml.AvaloniaRuntimeXamlLoader.Load(string.Format("<Span xml:space=\"preserve\" xmlns=\"https://github.com/avaloniaui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2006/xaml\">{0}</Span>", text)) is Span result)
                    {
                        textBlock.Inlines?.Clear();
                        textBlock.Inlines?.Add(result);
                        textBlock.InvalidateVisual();
                    }
                }
            });
        }
    }
}
