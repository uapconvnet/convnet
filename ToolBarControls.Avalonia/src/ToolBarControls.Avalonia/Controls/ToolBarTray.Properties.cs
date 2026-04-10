using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Metadata;

namespace ToolBarControls.Avalonia;

public partial class ToolBarTray
{
    #region Background Property

    /// <summary>
    /// Defines the <see cref="Background"/> Property
    /// </summary>
    public static readonly StyledProperty<IBrush?> BackgroundProperty =
        Border.BackgroundProperty.AddOwner<ToolBarTray>();

    /// <summary>
    /// Get or set background
    /// </summary>
    public IBrush? Background
    {
        get => GetValue(BackgroundProperty);
        set => SetValue(BackgroundProperty, value);
    }

    #endregion

    #region Orientation Property

    /// <summary>
    /// Defines the <see cref="Orientation"/> Property
    /// </summary>
    public static readonly StyledProperty<Orientation> OrientationProperty =
        StackPanel.OrientationProperty.AddOwner<ToolBarTray>(
            new StyledPropertyMetadata<Orientation>(Orientation.Horizontal));

    /// <summary>
    /// Get or set the orientation
    /// </summary>
    public Orientation Orientation
    {
        get => GetValue(OrientationProperty);
        set => SetValue(OrientationProperty, value);
    }

    private static void OnOrientationChanged(ToolBarTray toolBar, AvaloniaPropertyChangedEventArgs e)
    {
        var toolbarCollection = toolBar.ToolBars;
        foreach (var t in toolbarCollection)
        {
            t.CoerceValue(ToolBar.OrientationProperty);
        }
    }

    #endregion

    #region IsLocked Property

    /// <summary>
    /// Defines the <see cref="IsLocked"/> Property
    /// </summary>
    public static readonly StyledProperty<bool> IsLockedProperty =
        AvaloniaProperty.Register<ToolBarTray, bool>(nameof(IsLocked), inherits: true);

    /// <summary>
    /// Get or set lock
    /// </summary>
    public bool IsLocked
    {
        get => GetValue(IsLockedProperty);
        set => SetValue(IsLockedProperty, value);
    }

    #endregion

    #region ToolBars Property

    /// <summary>
    /// Collection of ToolBar
    /// </summary>
    [Content]
    public Collection<ToolBar> ToolBars
    {
        get => _toolBarsCollection ??= new ToolBarCollection(this);
    }

    #endregion

    static ToolBarTray()
    {
        ToolBarThumb.DragDeltaEvent.AddClassHandler<ToolBarTray>(OnThumbDragDelta);
        OrientationProperty.Changed.AddClassHandler<ToolBarTray>(OnOrientationChanged);

        AffectsRender<ToolBarTray>(BackgroundProperty);
        AffectsMeasure<ToolBarTray>(IsLockedProperty);
    }

    private static void OnThumbDragDelta(ToolBarTray toolBarTray, ToolBarVectorEventArgs e)
    {
        // Don't move toolbars if IsLocked == true
        if (toolBarTray.IsLocked)
            return;

        toolBarTray.ProcessThumbDragDelta(e);
    }
}