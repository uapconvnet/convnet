using System.Collections.ObjectModel;
using Avalonia.Layout;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ToolBarControls.Avalonia.Demo.ViewModels.ToolBars;

public partial class ToolBarViewModel : ViewModelBase
{
    [ObservableProperty] private int _band;
    [ObservableProperty] private int _bandIndex;
    [ObservableProperty] private string? _header;
    [ObservableProperty] private bool _isEnabled = true;
    [ObservableProperty] private Orientation _orientation = Orientation.Horizontal;
    [ObservableProperty] private ObservableCollection<object> _items = new();
}