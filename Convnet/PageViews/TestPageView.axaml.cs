using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace Convnet.PageViews
{
    public partial class TestPageView : UserControl
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public TestPageView()
        {
            //string[] names = this.GetType().Assembly.GetManifestResourceNames();
            //string[] anames = Assembly.GetExecutingAssembly().GetManifestResourceNames();
           
            InitializeComponent();


            //var gr = this.FindControl<Grid>("grid");
        }
      
        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
