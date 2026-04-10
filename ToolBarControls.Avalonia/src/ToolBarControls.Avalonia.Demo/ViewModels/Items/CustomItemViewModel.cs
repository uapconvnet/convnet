using CommunityToolkit.Mvvm.ComponentModel;

namespace ToolBarControls.Avalonia.Demo.ViewModels.Items;

public partial class CustomItemViewModel : ViewModelBase
{
    [ObservableProperty] private string _text;
    [ObservableProperty] private string _buttonContent;

    public CustomItemViewModel(int number)
    {
        _text = $"{number}+{number} =";
        _buttonContent = (number + number).ToString();
    }
}