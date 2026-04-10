using Avalonia.Layout;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ToolBarControls.Avalonia.Demo.ViewModels.ToolBars;

public partial class ToolBarTrayViewModel : ViewModelBase
{
    [ObservableProperty] private bool _isEnabled = true;
    [ObservableProperty] private bool _isLocked;
    [ObservableProperty] private Orientation _orientation = Orientation.Horizontal;
}