using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Styling;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;


namespace Convnet.Common
{
    public class FormattedTextBlock : TextBlock, INotifyPropertyChanged, IStyleable
    {
        Type IStyleable.StyleKey => typeof(TextBlock);

        public new event PropertyChangedEventHandler? PropertyChanged;

        //public static readonly StyledProperty<string> FormattedTextProperty = AvaloniaProperty.Register<FormattedTextBlock, string>(nameof(FormattedText), defaultValue: string.Empty, false, Avalonia.Data.BindingMode.TwoWay);

        public static readonly DirectProperty<FormattedTextBlock, string?> FormattedTextProperty = AvaloniaProperty.RegisterDirect<FormattedTextBlock, string?>(
          nameof(FormattedText),
          o => o.FormattedText,
          (o, v) =>
          {
              if (string.Compare(o.FormattedText, v) != 0)
                  o.FormattedText = v;
          },
          string.Empty,
          Avalonia.Data.BindingMode.OneWay);

        public string? FormattedText
        {
            get { return base.Text; }
            set
            {
                if (value != base.Text)
                {
                    string? formattedText = (string?)value ?? string.Empty;
                    formattedText = string.Format("<Span xml:space=\"preserve\" xmlns=\"https://github.com/avaloniaui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2006/xaml\">{0}</Span>", formattedText);

                    using (TextReader sr = new StringReader(formattedText))
                    {
                        if (Avalonia.Markup.Xaml.AvaloniaRuntimeXamlLoader.Load(sr.ReadToEnd()) is Span result)
                        {
                            Inlines?.Clear();
                            Inlines?.Add(result);
                        }
                    }

                    //base.Text = value;
                    //SetValue<string?>(FormattedTextProperty, value);
                    OnPropertyChanged(nameof(FormattedText));
                    //OnPropertyChanged(nameof(Text));
                }
            }
        }

        public FormattedTextBlock()
        {
            
        }
  


        #region INotifyPropertyChanged Members

        protected virtual void OnPropertyChanged(string propertyName)
        {
            this.VerifyPropertyName(propertyName);

            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #region Debugging Aides

        /// <summary>
        /// Warns the developer if this object does not have
        /// a public property with the specified name. This 
        /// method does not exist in a Release build.
        /// </summary>
        [Conditional("DEBUG")]
        [DebuggerStepThrough]
        public void VerifyPropertyName(string propertyName)
        {
            // If you raise PropertyChanged and do not specify a property name,
            // all properties on the object are considered to be changed by the binding system.
            if (String.IsNullOrEmpty(propertyName))
                return;

            // Verify that the property name matches a real,  
            // public, instance property on this object.
            if (TypeDescriptor.GetProperties(this)[propertyName] == null)
            {
                string msg = "Invalid property name: " + propertyName;

                if (this.ThrowOnInvalidPropertyName)
                    throw new ArgumentException(msg);
                else
                    Debug.Fail(msg);
            }
        }

        /// <summary>
        /// Returns whether an exception is thrown, or if a Debug.Fail() is used
        /// when an invalid property name is passed to the VerifyPropertyName method.
        /// The default value is false, but subclasses used by unit tests might 
        /// override this property's getter to return true.
        /// </summary>
        protected virtual bool ThrowOnInvalidPropertyName { get; private set; }

        #endregion // Debugging Aides

        #endregion // INotifyPropertyChanged Members
    }
}
