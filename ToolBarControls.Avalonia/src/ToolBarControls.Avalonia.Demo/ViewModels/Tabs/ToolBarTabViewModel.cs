using System.Collections.ObjectModel;
using Avalonia.Layout;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ToolBarControls.Avalonia.Demo.Helpers;
using ToolBarControls.Avalonia.Demo.ViewModels.ToolBars;

namespace ToolBarControls.Avalonia.Demo.ViewModels.Tabs;

public partial class ToolBarTabViewModel : ViewModelBase
{
    [ObservableProperty] private string _pressedElementName = string.Empty;

    public ToolBarTabViewModel()
    {
        #region ToolBarViewModel1 Init

        ToolBarViewModel1 = new ToolBarViewModel
        {
            Items = new ObservableCollection<object>(ControlHelper.CreateDefaultItems()),
            Orientation = Orientation.Horizontal
        };

        _isVertical1 = ToolBarViewModel1.Orientation == Orientation.Vertical;
        _isEnabled1 = ToolBarViewModel1.IsEnabled;

        #endregion

        #region ToolBarViewModel2 Init

        ToolBarViewModel2 = new ToolBarViewModel
        {
            Items = new ObservableCollection<object>(ControlHelper.CreateDefaultItemViewModels()),
            Orientation = Orientation.Horizontal
        };

        _isVertical2 = ToolBarViewModel2.Orientation == Orientation.Vertical;
        _isEnabled2 = ToolBarViewModel2.IsEnabled;

        #endregion
    }

    #region ToolBarViewModel1

    [ObservableProperty] private bool _isVertical1;
    [ObservableProperty] private bool _isEnabled1;

    public ToolBarViewModel ToolBarViewModel1 { get; }

    [RelayCommand]
    private void AddNewButton()
    {
        ToolBarViewModel1.Items.Add(ControlHelper.CreateButton(ToolBarViewModel1.Items.Count));
    }

    [RelayCommand]
    private void AddNewToggle()
    {
        ToolBarViewModel1.Items.Add(ControlHelper.CreateToggleButton(ToolBarViewModel1.Items.Count));
    }

    [RelayCommand]
    private void AddNewMenu()
    {
        ToolBarViewModel1.Items.Add(ControlHelper.CreateMenu(ToolBarViewModel1.Items.Count));
    }

    [RelayCommand(CanExecute = nameof(CanRemoveItem))]
    private void RemoveItem()
    {
        ToolBarViewModel1.Items.RemoveAt(ToolBarViewModel1.Items.Count - 1);
    }

    private bool CanRemoveItem()
    {
        return ToolBarViewModel1.Items.Count != 0;
    }

    partial void OnIsVertical1Changed(bool value)
    {
        ToolBarViewModel1.Orientation = value ? Orientation.Vertical : Orientation.Horizontal;
    }

    partial void OnIsEnabled1Changed(bool value)
    {
        ToolBarViewModel1.IsEnabled = value;
    }

    #endregion

    #region ToolBarViewModel2

    [ObservableProperty] private bool _isVertical2;
    [ObservableProperty] private bool _isEnabled2;

    public ToolBarViewModel ToolBarViewModel2 { get; }

    [RelayCommand]
    private void AddNewViewModelItem()
    {
        ToolBarViewModel2.Items.Add(ControlHelper.CreateCustomItemViewModel(ToolBarViewModel2.Items.Count));
    }

    [RelayCommand(CanExecute = nameof(CanRemoveViewModelItem))]
    private void RemoveViewModelItem()
    {
        ToolBarViewModel2.Items.RemoveAt(ToolBarViewModel2.Items.Count - 1);
    }

    private bool CanRemoveViewModelItem()
    {
        return ToolBarViewModel2.Items.Count != 0;
    }

    partial void OnIsVertical2Changed(bool value)
    {
        ToolBarViewModel2.Orientation = value ? Orientation.Vertical : Orientation.Horizontal;
    }

    partial void OnIsEnabled2Changed(bool value)
    {
        ToolBarViewModel2.IsEnabled = value;
    }

    #endregion

    [RelayCommand]
    private void PressedElement(string elementName)
    {
        PressedElementName = $"Pressed Element: {elementName}";
    }
}