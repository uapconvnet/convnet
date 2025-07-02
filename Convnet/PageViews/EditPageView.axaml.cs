using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using AvaloniaEdit.Highlighting;
using AvaloniaEdit.Highlighting.Xshd;
using AvaloniaEdit.Indentation.CSharp;
using AvaloniaEdit.TextMate;
using Convnet.Common;
using Convnet.PageViewModels;
using Convnet.Properties;
using System;
using System.IO;
using System.Linq;
using System.Xml;
using TextMateSharp.Grammars;

namespace Convnet.PageViews
{
    public partial class EditPageView : UserControl
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public EditPageView()
        {
            InitializeComponent();

            IHighlightingDefinition DefinitionHighlighting;
            using (Stream? s = typeof(EditPageView).Assembly.GetManifestResourceStream("Convnet.Resources.Definition.xshd"))
            {
                if (s == null)
                    throw new InvalidOperationException("Could not find embedded resource");
                using (XmlReader reader = new XmlTextReader(s))
                {
                    DefinitionHighlighting = HighlightingLoader.Load(reader, HighlightingManager.Instance);
                }
            }
            HighlightingManager.Instance.RegisterHighlighting("Definition", [".txt"], DefinitionHighlighting);
            var editorDefinition = this.FindControl<CodeEditor>("EditorDefinition");
            if (editorDefinition != null)
            {
                editorDefinition.SyntaxHighlighting = HighlightingManager.Instance.GetDefinitionByExtension(".txt");
                editorDefinition.TextChanged += EditorDefinition_TextChanged;
                
                /* var line = editorDefinition.Document.GetLineByNumber(Settings.Default.LineDefinition);
                editorDefinition.CaretOffset = line.Offset + Settings.Default.ColumnDefinition;
                editorDefinition.TextArea.Caret.BringCaretToView(); // ← Try this call. */
            }

            //IHighlightingDefinition CSharpHighlighting;
            //using (Stream? s = typeof(EditPageView).Assembly.GetManifestResourceStream("Convnet.Resources.CSharp-Mode.xshd"))
            //{
            //    if (s == null)
            //        throw new InvalidOperationException("Could not find embedded resource");
            //    using (XmlReader reader = new XmlTextReader(s))
            //    {
            //        CSharpHighlighting = HighlightingLoader.Load(reader, HighlightingManager.Instance);
            //    }
            //}
            //HighlightingManager.Instance.RegisterHighlighting("C#", [".cs"], CSharpHighlighting);
            //var editorScript = this.FindControl<CodeEditor>("EditorScript");
            //if (editorScript != null)
            //{
            //    editorScript.SyntaxHighlighting = HighlightingManager.Instance.GetDefinitionByExtension(".cs");
            //    editorScript.TextChanged += EditorScript_TextChanged;
            //}

            var editorScript = this.FindControl<CodeEditor>("EditorScript");
            if (editorScript != null)
            {
                editorScript.SyntaxHighlighting = HighlightingManager.Instance.GetDefinitionByExtension(".cs");
                editorScript.TextChanged += EditorScript_TextChanged;
                editorScript.TextArea.IndentationStrategy = new CSharpIndentationStrategy(editorScript.Options);

                var registryOptions = new RegistryOptions(ThemeName.DarkPlus);
                var textMateInstallation = editorScript.InstallTextMate(registryOptions);
                var csharpLanguage = registryOptions.GetLanguageByExtension(".cs");
                textMateInstallation.SetGrammar(registryOptions.GetScopeByLanguageId(csharpLanguage.Id));

                /* var line = editorScript.Document.GetLineByNumber(Settings.Default.LineScript);
                editorScript.CaretOffset = line.Offset + Settings.Default.ColumnScript;
                editorScript.TextArea.Caret.BringCaretToView(); // ← Try this call. */

            }

            var gr = this.FindControl<Grid>("grid");
            if (gr != null)
                gr.ColumnDefinitions.First().Width = new GridLength(Settings.Default.EditSplitPositionA, GridUnitType.Pixel);
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private void EditorDefinition_TextChanged(object? sender, EventArgs e)
        {
            if (DataContext != null && sender != null)
            {
                var epvm = DataContext as EditPageViewModel;
                if (epvm != null)
                    epvm.Definition = ((CodeEditor)sender).Text;
            }
        }

        private void EditorScript_TextChanged(object? sender, EventArgs e)
        {
            if (DataContext != null && sender != null)
            {
                var epvm = DataContext as EditPageViewModel;
                if (epvm != null)
                    epvm.Script = ((CodeEditor)sender).Text;
            }
        }

        public void GridSplitter_DragCompleted(object? sender, VectorEventArgs e)
        {
            if (!e.Handled)
            {
                var gr = this.FindControl<Grid>("grid");
                if (gr != null)
                { 
                    Settings.Default.EditSplitPositionA = gr.ColumnDefinitions.First().ActualWidth;
                    Settings.Default.Save();
                    e.Handled = true;
                }
            }
        }
    }
}
