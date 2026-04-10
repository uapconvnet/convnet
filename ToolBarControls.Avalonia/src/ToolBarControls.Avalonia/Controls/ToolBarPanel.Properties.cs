using Avalonia;
using Avalonia.Controls;

namespace ToolBarControls.Avalonia;

public partial class ToolBarPanel
{
    #region ItemIsOwnContainer Property

    private static readonly AttachedProperty<bool> ItemIsOwnContainerProperty =
        AvaloniaProperty.RegisterAttached<ToolBar, Control, bool>("ItemIsOwnContainer");

    #endregion
}