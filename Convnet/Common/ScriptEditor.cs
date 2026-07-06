using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Immutable;
using Avalonia.Threading;
using AvaloniaEdit;
using AvaloniaEdit.Document;
using AvaloniaEdit.Editing;
using AvaloniaEdit.Folding;
using AvaloniaEdit.Rendering;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Convnet.Common
{
    internal sealed class CustomMargin : AbstractMargin
    {
        private readonly IBrush _defaultbackgroundBrush = Brushes.Transparent;
        private IBrush _backgroundBrush = new ImmutableSolidColorBrush(new Color(255, 51, 51, 51));
        private readonly IBrush _pointerOverBrush = new ImmutableSolidColorBrush(new Color(192, 80, 80, 80));
        private readonly IPen _pointerOverPen = new ImmutablePen(new ImmutableSolidColorBrush(new Color(192, 37, 37, 37)), 1);
        private readonly IBrush _markerBrush = new ImmutableSolidColorBrush(new Color(255, 195, 81, 92));
        private readonly IPen _markerPen = new ImmutablePen(new ImmutableSolidColorBrush(new Color(255, 240, 92, 104)), 1);

        private readonly List<int> _markedDocumentLines = [];
        private int _pointerOverLine = -1;

        public IBrush BackGroundBrush
        {
            get => _backgroundBrush;
            set
            {
                _backgroundBrush = value;
                InvalidateVisual();
            }
        }

        public void SetDefaultBackgroundBrush()
        {
            _backgroundBrush = _defaultbackgroundBrush;
            InvalidateVisual();
        }

        public CustomMargin()
        {
            Cursor = new Cursor(StandardCursorType.Arrow);
        }

        protected override void OnTextViewChanged(TextView? oldTextView, TextView? newTextView)
        {
            if (oldTextView != null)
            {
                oldTextView.VisualLinesChanged -= OnVisualLinesChanged;
                oldTextView.DocumentChanged -= OnDocumentChanged;
            }

            if (newTextView != null)
            {
                newTextView.VisualLinesChanged += OnVisualLinesChanged;
                newTextView.DocumentChanged += OnDocumentChanged;
            }

            base.OnTextViewChanged(oldTextView, newTextView);
        }

        private void OnVisualLinesChanged(object? sender, EventArgs eventArgs)
        {
            InvalidateVisual();
        }

        private void OnDocumentChanged(object? sender, DocumentChangedEventArgs e)
        {
            _markedDocumentLines.Clear();
            InvalidateVisual();
        }

        protected override Size MeasureOverride(Size availableSize)
        {
            return new Size(20, 0);
        }

        private int GetLineNumber(PointerEventArgs e)
        {
            double visualY = e.GetPosition(TextView).Y + TextView.VerticalOffset;
            VisualLine visualLine = TextView.GetVisualLineFromVisualTop(visualY);
            return (visualLine == null) ? -1 : visualLine.FirstDocumentLine.LineNumber;
        }

        protected override void OnPointerMoved(PointerEventArgs e)
        {
            _pointerOverLine = GetLineNumber(e);
            InvalidateVisual();

            base.OnPointerMoved(e);
        }

        protected override void OnPointerExited(PointerEventArgs e)
        {
            _pointerOverLine = -1;
            InvalidateVisual();

            base.OnPointerExited(e);
        }

        protected override void OnPointerPressed(PointerPressedEventArgs e)
        {
            int line = _pointerOverLine = GetLineNumber(e);

            if (!_markedDocumentLines.Remove(line))
                _markedDocumentLines.Add(line);

            _markedDocumentLines.Sort();
            InvalidateVisual();
            e.Handled = true;

            base.OnPointerPressed(e);
        }

        public override void Render(DrawingContext context)
        {
            context.DrawRectangle(_backgroundBrush, null, Bounds);

            if (TextView?.VisualLinesValid == true)
            {
                foreach (var visualLine in TextView.VisualLines)
                {
                    double y = visualLine.VisualTop - TextView.VerticalOffset + visualLine.Height / 2;

                    if (_markedDocumentLines.Contains(visualLine.FirstDocumentLine.LineNumber))
                        context.DrawEllipse(_markerBrush, _markerPen, new Point(10, y), 8, 8);
                    else if (_pointerOverLine == visualLine.FirstDocumentLine.LineNumber)
                        context.DrawEllipse(_pointerOverBrush, _pointerOverPen, new Point(10, y), 8, 8);
                }
            }

            base.Render(context);
        }
    }
    
     public class CSharpFoldingStrategy
     {
        public void UpdateFoldings(FoldingManager manager, TextDocument document)
        {
            var foldings = CreateNewFoldings(document, out var firstErrorOffset);
            manager.UpdateFoldings(foldings, firstErrorOffset);
        }

        public IEnumerable<NewFolding> CreateNewFoldings(TextDocument document, out int firstErrorOffset)
        {
            firstErrorOffset = -1;
            var newFoldings = new List<NewFolding>();
            var startOffsets = new Stack<int>();

            for (int offset = 0; offset < document.TextLength; offset++)
            {
                char c = document.GetCharAt(offset);
                switch (c)
                {
                    case '{':
                        startOffsets.Push(offset);
                        break;
                    case '}':
                        if (startOffsets.Count > 0)
                        {
                            int startOffset = startOffsets.Pop();
                            // Add a folding if the block spans more than one line
                            int startLine = document.GetLineByOffset(startOffset).LineNumber;
                            int endLine = document.GetLineByOffset(offset).LineNumber;
                            if (startLine < endLine)
                            {
                                newFoldings.Add(new NewFolding(startOffset, offset + 1));
                            }
                        }
                        break;
                }
            }

            newFoldings.Sort((a, b) => a.StartOffset.CompareTo(b.StartOffset));
            return newFoldings;
        }
    }
 
    public class ScriptEditor : TextEditor, INotifyPropertyChanged
    {
        protected override Type StyleKeyOverride => typeof(TextEditor);

        public new event PropertyChangedEventHandler? PropertyChanged;

        private readonly FoldingManager foldingManager;
        private readonly CSharpFoldingStrategy foldingStrategy;
        //private CustomMargin customMargin;

        public ScriptEditor()
        {
            LineNumbersMargin = new Thickness(4,0,0,0); 

            FontSize = 14;
            FontFamily = new FontFamily("Cascadia Code,Consolas,Menlo,Monospace");

            foldingStrategy = new CSharpFoldingStrategy();

            Options = new TextEditorOptions
            {
                IndentationSize = 4,
                ConvertTabsToSpaces = false,
                AllowScrollBelowDocument = true,
                HighlightCurrentLine = true,
                EnableHyperlinks = true,
                EnableEmailHyperlinks = true,
                AcceptsTab = true
                
            };
            TextArea.IndentationStrategy = new AvaloniaEdit.Indentation.CSharp.CSharpIndentationStrategy(Options);
            TextArea.RightClickMovesCaret = true;
          
            //customMargin = new CustomMargin();
            //TextArea.LeftMargins.Insert(0, customMargin);

            foldingManager = FoldingManager.Install(TextArea);
            foldingStrategy.UpdateFoldings(foldingManager, Document);
            TextChanged += (sender, args) => foldingStrategy.UpdateFoldings(foldingManager, Document);
            
            var cmdKey = ApplicationHelper.GetPlatformCommandKey();

            var cm = new ContextMenu();

            var cut = new MenuItem { Header = "Cut", InputGesture = new KeyGesture(Key.X, cmdKey) };
            var copy = new MenuItem { Header = "Copy", InputGesture = new KeyGesture(Key.C, cmdKey) };
            var paste = new MenuItem { Header = "Paste", InputGesture = new KeyGesture(Key.V, cmdKey) };
            var delete = new MenuItem { Header = "Delete", InputGesture = new KeyGesture(Key.Delete) };
            var selectAll = new MenuItem { Header = "Select All", InputGesture = new KeyGesture(Key.A, cmdKey) };
            var undo = new MenuItem { Header = "Undo", InputGesture = new KeyGesture(Key.Z, cmdKey) };
            var redo = new MenuItem { Header = "Redo", InputGesture = new KeyGesture(Key.Y, cmdKey) };

            cut.Icon = ApplicationHelper.LoadFromResource("Cut.png");
            paste.Icon = ApplicationHelper.LoadFromResource("Paste.png");
            copy.Icon = ApplicationHelper.LoadFromResource("Copy.png");
            delete.Icon = ApplicationHelper.LoadFromResource("Cancel.png");
            selectAll.Icon = ApplicationHelper.LoadFromResource("SelectAll.png");
            undo.Icon = ApplicationHelper.LoadFromResource("Undo.png");
            redo.Icon = ApplicationHelper.LoadFromResource("Redo.png");

            cut.Command = ApplicationCommands.Cut;
            paste.Command = ApplicationCommands.Paste;
            copy.Command = ApplicationCommands.Copy;
            delete.Command = ApplicationCommands.Delete;
            selectAll.Command = ApplicationCommands.SelectAll;
            undo.Command = ApplicationCommands.Undo;
            redo.Command = ApplicationCommands.Redo;

            cut.Click += (s, e) => { if (CanCut) Dispatcher.UIThread.Post(() => Cut()); };
            paste.Click += (s, e) => { if (CanPaste) Dispatcher.UIThread.Post(() => Paste()); };
            copy.Click += (s, e) => { if (CanCopy) Dispatcher.UIThread.Post(() => Copy()); };
            delete.Click += (s, e) => { if (CanDelete) Dispatcher.UIThread.Post(() => Delete()); };
            selectAll.Click += (s, e) => { if (CanSelectAll) Dispatcher.UIThread.Post(() => SelectAll()); };
            undo.Click += (s, e) => { if (CanUndo) Dispatcher.UIThread.Post(() => Undo()); };
            redo.Click += (s, e) => { if (CanRedo) Dispatcher.UIThread.Post(() => Redo()); };

            cm.Items.Add(cut);
            cm.Items.Add(copy);
            cm.Items.Add(paste);
            cm.Items.Add(delete);
            cm.Items.Add(new Separator
            {
                Width = Double.NaN,
                Height = 1,
                Margin = new Thickness(1)
            });
            cm.Items.Add(selectAll);
            cm.Items.Add(new Separator
            {
                Width = Double.NaN,
                Height = 1,
                Margin = new Thickness(1)
            });
            cm.Items.Add(undo);
            cm.Items.Add(redo);

            ContextMenu = cm;
        }

        public static readonly DirectProperty<ScriptEditor, string> ScriptProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, string>(
            nameof(Script),
            o => o.Script,
            (o, v) =>
            {
                if (string.Compare(o.Script, v) != 0)
                    o.Script = v;
            },
            "",
            Avalonia.Data.BindingMode.TwoWay);

        public string Script
        {
            get { return base.Text; }
            set
            {
                if (value != base.Text)
                {
                    base.Text = value;
                    SetValue(ScriptProperty, value);
                    OnPropertyChanged(nameof(Script));
                }
            }
        }

        protected override void OnTextChanged(EventArgs e)
        {
            //SetCurrentValue(ScriptProperty, base.Text);
            base.OnTextChanged(e);
            OnPropertyChanged(nameof(Length));
        }

        public int Length
        {
            get { return base.Text.Length; }
        }

        public static readonly DirectProperty<ScriptEditor, TextLocation> TextLocationProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, TextLocation>(
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
            get => base.Document.GetLocation(SelectionStart);
            set
            {
                if (value.Line <= Document.LineCount && GetValue<TextLocation>(TextLocationProperty) != value)
                    Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(() => { 
                        TextArea.Caret.Line = value.Line; 
                        TextArea.Caret.Column = value.Column; 
                        TextArea.Caret.BringCaretToView();
                        if (this.IsFocused)
                        {
                            TextArea.Caret.Show();
                        }
                        ScrollTo(value.Line, value.Column); 
                        OnPropertyChanged(nameof(TextLocation)); }, DispatcherPriority.ContextIdle);
            }
        }

        public static readonly DirectProperty<ScriptEditor, double> VerticalOffsetProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, double>(
                nameof(VerticalOffset),
                o => o.VerticalOffset,
                (o, v) =>
                {
                    if (!o.VerticalOffset.Equals(v))
                        o.VerticalOffset = v;
                },
                0,
                Avalonia.Data.BindingMode.TwoWay);

        public new double VerticalOffset
        {
            get => base.VerticalOffset;
            set
            {
                if (base.VerticalOffset != value)
                    Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(() => { this.ScrollToVerticalOffset(value); OnPropertyChanged(nameof(VerticalOffset));}, DispatcherPriority.ContextIdle);
            }
        }

        public static readonly DirectProperty<ScriptEditor, int> CaretOffsetProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, int>(
            nameof(CaretOffset),
            o => o.CaretOffset,
            (o, v) =>
            {
                if (o.CaretOffset != v)
                    o.CaretOffset = v;
            },
            0,
            Avalonia.Data.BindingMode.TwoWay);

        public new int CaretOffset
        {
            get => base.CaretOffset;
            set
            {
                if (base.CaretOffset != value)
                {
                    base.CaretOffset = value;
                    OnPropertyChanged(nameof(CaretOffset));
                }
            }
        }

        public static readonly DirectProperty<ScriptEditor, int> SelectionLengthProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, int>(
            nameof(SelectionLength),
            o => o.SelectionLength,
            (o, v) =>
            {
                if (o.SelectionLength != v)
                    o.SelectionLength = v;
            },
            0,
            Avalonia.Data.BindingMode.TwoWay);

        public new int SelectionLength
        {
            get { return base.SelectionLength; }
            set
            {
                if (base.SelectionLength != value)
                {
                    base.SelectionLength = value;
                    OnPropertyChanged(nameof(SelectionLength));
                }
            }
        }

        public static readonly DirectProperty<ScriptEditor, int> SelectionStartProperty = AvaloniaProperty.RegisterDirect<ScriptEditor, int>(
            nameof(SelectionStart),
            o => o.SelectionStart,
            (o, v) =>
            {
                if (o.SelectionStart != v)
                    o.SelectionStart = v;
            },
            0,
            Avalonia.Data.BindingMode.TwoWay);

        public new int SelectionStart
        {
            get => base.SelectionStart;
            set 
            {
                if (base.SelectionStart != value)
                {
                    base.SelectionStart = value;
                    OnPropertyChanged(nameof(SelectionStart));
                }
            }
        }

        public static object? VisualLine { get; private set; }

        //public static readonly StyledProperty<string> FilePathProperty = AvaloniaProperty.Register<ScriptEditor, string>(nameof(FilePath), defaultValue: string.Empty, false, Avalonia.Data.BindingMode.TwoWay);

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

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
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
        public void VerifyPropertyName([CallerMemberName] string? propertyName = null)
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
