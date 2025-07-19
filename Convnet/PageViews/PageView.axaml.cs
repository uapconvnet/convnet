using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;

using Convnet.PageViewModels;

namespace Convnet.PageViews
{
    public partial class PageView : UserControl
    {
        public PageView()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}