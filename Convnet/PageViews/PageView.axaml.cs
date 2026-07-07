using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
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

        private void CmdToolBar_GotFocus(object? sender, Avalonia.Input.FocusChangedEventArgs e)
        {
            var view =  this.FindControl<ComboBox>("PageViews");
            if (view != null) 
            {
                if (view.SelectedIndex == 0) // EditPageView
                {
                    Dispatcher.UIThread.Post(() => e.OldFocusedElement?.Focus());
                }
            }
        }
    }
}