using Avalonia.Markup.Xaml;
using Convnet.PageViewModels;
using ReactiveUI.Avalonia;

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