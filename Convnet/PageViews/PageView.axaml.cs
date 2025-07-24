using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;

using Convnet.PageViewModels;

namespace Convnet.PageViews
{
    public partial class PageView : ReactiveUserControl<PageViewModel>
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