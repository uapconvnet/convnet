using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Threading;
using AvaloniaEdit;
using AvaloniaEdit.Document;
using AvaloniaEdit.Editing;
using System;
using System.ComponentModel;
using System.Diagnostics;

namespace Convnet.Common
{
    public class DefinitionEditor : TextEditor, INotifyPropertyChanged
    {
        protected override Type StyleKeyOverride => typeof(TextEditor);

        public new event PropertyChangedEventHandler? PropertyChanged;

        public DefinitionEditor()
        {
            FontSize = 14;
            FontFamily = new FontFamily("Cascadia Code,Consolas,Menlo,Monospace");
            Options = new TextEditorOptions
            {
                IndentationSize = 4,
                ConvertTabsToSpaces = false,
                AllowScrollBelowDocument = true,
                HighlightCurrentLine = true,
            };
            TextArea.RightClickMovesCaret = true;
            TextArea.IndentationStrategy = new AvaloniaEdit.Indentation.CSharp.CSharpIndentationStrategy(Options);
          
            var cmdKey = ApplicationHelper.GetPlatformCommandKey();

            var cm = new ContextMenu();

            var cut = new MenuItem { Header = "Cut", InputGesture = new KeyGesture(Key.X, cmdKey) };
            var copy = new MenuItem { Header = "Copy", InputGesture = new KeyGesture(Key.C, cmdKey) };
            var paste = new MenuItem { Header = "Paste", InputGesture = new KeyGesture(Key.V, cmdKey) };
            var delete = new MenuItem { Header = "Delete", InputGesture = new KeyGesture(Key.Delete) };
            var selectall = new MenuItem { Header = "Select All", InputGesture = new KeyGesture(Key.A, cmdKey) };
            var undo = new MenuItem { Header = "Undo", InputGesture = new KeyGesture(Key.Z, cmdKey) };
            var redo = new MenuItem { Header = "Redo", InputGesture = new KeyGesture(Key.Y, cmdKey) };

            cut.Icon = ApplicationHelper.LoadFromResource("Cut.png");
            paste.Icon = ApplicationHelper.LoadFromResource("Paste.png");
            copy.Icon = ApplicationHelper.LoadFromResource("Copy.png");
            delete.Icon = ApplicationHelper.LoadFromResource("Cancel.png");
            selectall.Icon = ApplicationHelper.LoadFromResource("SelectAll.png");
            undo.Icon = ApplicationHelper.LoadFromResource("Undo.png");
            redo.Icon = ApplicationHelper.LoadFromResource("Redo.png");

            cut.Command = ApplicationCommands.Cut;
            paste.Command = ApplicationCommands.Paste;
            copy.Command = ApplicationCommands.Copy;
            delete.Command = ApplicationCommands.Delete;
            selectall.Command = ApplicationCommands.SelectAll;
            undo.Command = ApplicationCommands.Undo;
            redo.Command = ApplicationCommands.Redo;

            cut.Click += (s, e) => { if (CanCut) Dispatcher.UIThread.Post(() => Cut()); };
            paste.Click += (s, e) => { if (CanPaste) Dispatcher.UIThread.Post(() => Paste()); };
            copy.Click += (s, e) => { if (CanCopy) Dispatcher.UIThread.Post(() => Copy()); };
            delete.Click += (s, e) => { if (CanDelete) Dispatcher.UIThread.Post(() => Delete()); };
            selectall.Click += (s, e) => { Dispatcher.UIThread.Post(() => SelectAll()); };
            undo.Click += (s, e) => { if (CanUndo) Dispatcher.UIThread.Post(() => Undo()); };
            redo.Click += (s, e) => { if (CanRedo) Dispatcher.UIThread.Post(() => Redo()); };

            cm.Items.Add(cut);
            cm.Items.Add(copy);
            cm.Items.Add(paste);
            cm.Items.Add(delete);
            cm.Items.Add(new Avalonia.Controls.Separator());
            cm.Items.Add(selectall);
            cm.Items.Add(new Avalonia.Controls.Separator());
            cm.Items.Add(undo);
            cm.Items.Add(redo);

            ContextMenu = cm;

            //TextChanged += DefinitionEditor_TextChanged;
            //LostFocus += DefinitionEditor_LostFocus;
        }

        //private void DefinitionEditor_LostFocus(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        //{
        //    Code = base.Text;
        //    SetValue(CodeProperty, Code);
        //    OnPropertyChanged(nameof(Code));
        //}

        //private void DefinitionEditor_TextChanged(object? sender, EventArgs e)
        //{
        //    Code = base.Text;
        //    SetValue(CodeProperty, Code);
        //    OnPropertyChanged(nameof(Code));
        //}

        public static readonly DirectProperty<DefinitionEditor, string> CodeProperty = AvaloniaProperty.RegisterDirect<DefinitionEditor, string>(
            nameof(Code),
            o => o.Code,
            (o, v) =>
            {
                if (string.Compare(o.Code, v) != 0)
                    o.Code = v;
            },
            "",
            Avalonia.Data.BindingMode.TwoWay);

        public string Code
        {
            get { return base.Text; }
            set
            {
                if (value != base.Text)
                {
                    base.Text = value;
                   SetValue(CodeProperty, value);
                    OnPropertyChanged(nameof(Code));
                }
            }
        }

        protected override void OnTextChanged(EventArgs e)
        {
            //SetCurrentValue(CodeProperty, base.Text);
            OnPropertyChanged(nameof(Length));
            base.OnTextChanged(e);
        }

        public int Length
        {
            get { return base.Text.Length; }
        }

        public static readonly DirectProperty<DefinitionEditor, TextLocation> TextLocationProperty = AvaloniaProperty.RegisterDirect<DefinitionEditor, TextLocation>(
           nameof(TextLocation),
           o => o.TextLocation,
           (o, v) =>
           {
               if (!o.TextLocation.Equals(v))
                   o.TextLocation = v;
           },
           new TextLocation(1, 1),
           Avalonia.Data.BindingMode.TwoWay);

        public TextLocation TextLocation
        {
            get { return base.Document.GetLocation(SelectionStart); }
            set
            {
                if (value.Line <= Document.LineCount && GetValue<TextLocation>(TextLocationProperty) != value)
                {
                    TextArea.Caret.Line = value.Line;
                    TextArea.Caret.Column = value.Column;
                    TextArea.Caret.BringCaretToView();
                    TextArea.Caret.Show();
                    ScrollTo(value.Line, value.Column);
                    SetValue(TextLocationProperty, value);
                    OnPropertyChanged(nameof(TextLocation));
                }
            }
        }

        //public static readonly DirectProperty<DefinitionEditor, int> CaretOffsetProperty = AvaloniaProperty.RegisterDirect<DefinitionEditor, int>(
        //    nameof(CaretOffset),
        //    o => o.CaretOffset,
        //    (o, v) =>
        //    {
        //        if (o.CaretOffset != v)
        //            o.CaretOffset = v;
        //    },
        //    0,
        //    Avalonia.Data.BindingMode.TwoWay);

        //public new int CaretOffset
        //{
        //    get { return base.CaretOffset; }
        //    set { SetValue<int>(CaretOffsetProperty, value); OnPropertyChanged(nameof(CaretOffset)); }
        //}

        //public static readonly DirectProperty<DefinitionEditor, int> SelectionLengthProperty = AvaloniaProperty.RegisterDirect<DefinitionEditor, int>(
        //    nameof(SelectionLength),
        //    o => o.SelectionLength,
        //    (o, v) =>
        //    {
        //        if (o.SelectionLength != v)
        //            o.SelectionLength = v;
        //    },
        //    0,
        //    Avalonia.Data.BindingMode.TwoWay);

        //public int SelectionLength
        //{
        //    get { return base.SelectionLength; }
        //    set { SetValue<int>(SelectionLengthProperty, value); OnPropertyChanged(nameof(SelectionLength)); }
        //}

        //public static readonly DirectProperty<DefinitionEditor, int> SelectionStartProperty = AvaloniaProperty.RegisterDirect<DefinitionEditor, int>(
        //    nameof(SelectionStart),
        //    o => o.SelectionStart,
        //    (o, v) =>
        //    {
        //        if (o.SelectionStart != v)
        //            o.SelectionStart = v;
        //    },
        //    0,
        //    Avalonia.Data.BindingMode.TwoWay);

        //public int SelectionStart
        //{
        //    get { return base.SelectionStart; }
        //    set { SetValue<int>(SelectionStartProperty, value); OnPropertyChanged(nameof(SelectionStart)); }
        //}

        public static object? VisualLine { get; private set; }

        //public static readonly StyledProperty<string> FilePathProperty = AvaloniaProperty.Register<DefinitionEditor, string>(nameof(FilePath), defaultValue: string.Empty, false, Avalonia.Data.BindingMode.TwoWay);

        //public string FilePath
        //{
        //    get { return GetValue(FilePathProperty); }
        //    set
        //    {
        //        SetValue(FilePathProperty, value);
        //        OnPropertyChanged(nameof(FilePath));
        //    }
        //}

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
