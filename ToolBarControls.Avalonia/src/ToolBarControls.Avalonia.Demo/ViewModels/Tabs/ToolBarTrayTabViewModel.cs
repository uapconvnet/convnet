using Avalonia.Layout;
using CommunityToolkit.Mvvm.ComponentModel;
using ToolBarControls.Avalonia.Demo.ViewModels.ToolBars;

namespace ToolBarControls.Avalonia.Demo.ViewModels.Tabs;

public partial class ToolBarTrayTabViewModel : ViewModelBase
{
    public ToolBarTrayTabViewModel()
    {
        ToolBarTrayViewModel1 = new ToolBarTrayViewModel();
        _isVertical1 = ToolBarTrayViewModel1.Orientation == Orientation.Vertical;
        _isEnabled1 = ToolBarTrayViewModel1.IsEnabled;
        _isLocked1 = ToolBarTrayViewModel1.IsLocked;

        ToolBarTrayViewModel2 = new ToolBarTrayViewModel();
        _isVertical2 = ToolBarTrayViewModel2.Orientation == Orientation.Vertical;
        _isEnabled2 = ToolBarTrayViewModel2.IsEnabled;
        _isLocked2 = ToolBarTrayViewModel2.IsLocked;
    }

    #region ToolBarTrayViewModel1

    [ObservableProperty] private bool _isVertical1;
    [ObservableProperty] private bool _isEnabled1;
    [ObservableProperty] private bool _isLocked1;

    public ToolBarTrayViewModel ToolBarTrayViewModel1 { get; }

    partial void OnIsVertical1Changed(bool value)
    {
        ToolBarTrayViewModel1.Orientation = value ? Orientation.Vertical : Orientation.Horizontal;
    }

    partial void OnIsEnabled1Changed(bool value)
    {
        ToolBarTrayViewModel1.IsEnabled = value;
    }

    partial void OnIsLocked1Changed(bool value)
    {
        ToolBarTrayViewModel1.IsLocked = value;
    }

    #endregion

    #region ToolBarTrayViewModel2

    [ObservableProperty] private bool _isVertical2;
    [ObservableProperty] private bool _isEnabled2;
    [ObservableProperty] private bool _isLocked2;

    public ToolBarTrayViewModel ToolBarTrayViewModel2 { get; }

    partial void OnIsVertical2Changed(bool value)
    {
        ToolBarTrayViewModel2.Orientation = value ? Orientation.Vertical : Orientation.Horizontal;
    }

    partial void OnIsEnabled2Changed(bool value)
    {
        ToolBarTrayViewModel2.IsEnabled = value;
    }

    partial void OnIsLocked2Changed(bool value)
    {
        ToolBarTrayViewModel2.IsLocked = value;
    }

    #endregion
}