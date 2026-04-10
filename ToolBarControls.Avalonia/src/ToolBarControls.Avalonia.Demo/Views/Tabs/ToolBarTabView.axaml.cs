using Avalonia.Controls;
using ToolBarControls.Avalonia.Demo.ViewModels.Tabs;

namespace ToolBarControls.Avalonia.Demo.Views.Tabs;

public partial class ToolBarTabView : UserControl
{
    public ToolBarTabView()
    {
        InitializeComponent();

        DataContext = new ToolBarTabViewModel();
    }
}