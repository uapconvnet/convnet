using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Threading;

namespace ToolBarControls.Avalonia;

public partial class ToolBar
{
    #region Orientation Property

    /// <summary>
    /// Defines the <see cref="Orientation"/> property.
    /// </summary>
    public static readonly StyledProperty<Orientation> OrientationProperty =
        AvaloniaProperty.Register<ToolBar, Orientation>(nameof(Orientation), inherits: true,
            coerce: CoerceOrientation);

    static Orientation CoerceOrientation(AvaloniaObject obj, Orientation value)
    {
        var toolBarTray = ((ToolBar)obj).ToolBarTray;
        return toolBarTray != null ? toolBarTray.Orientation : value;
    }

    /// <summary>
    /// Gets or sets the orientation of the ToolBar.
    /// </summary>
    public Orientation Orientation
    {
        get => GetValue(OrientationProperty);
        set => SetValue(OrientationProperty, value);
    }

    #endregion

    #region Band Property

    /// <summary>
    /// Defines the <see cref="Band"/> property.
    /// </summary>
    public static readonly StyledProperty<int>
        BandProperty = AvaloniaProperty.Register<ToolBar, int>(nameof(Band));

    /// <summary>
    /// Gets or sets the band number where ToolBar should be located withing the ToolBarTray.
    /// </summary>
    public int Band
    {
        get => GetValue(BandProperty);
        set => SetValue(BandProperty, value);
    }

    #endregion

    #region BandIndex Property

    /// <summary>
    /// Defines the <see cref="BandIndex"/> property.
    /// </summary>
    public static readonly StyledProperty<int> BandIndexProperty =
        AvaloniaProperty.Register<ToolBar, int>(nameof(BandIndex));

    /// <summary>
    /// Gets or sets the band index number where ToolBar should be located withing the band of ToolBarTray.
    /// </summary>
    public int BandIndex
    {
        get => GetValue(BandIndexProperty);
        set => SetValue(BandIndexProperty, value);
    }

    #endregion

    #region IsOverflowOpen Property

    /// <summary>
    /// Defines the <see cref="IsOverflowOpen"/> property.
    /// </summary>
    public static readonly StyledProperty<bool> IsOverflowOpenProperty =
        AvaloniaProperty.Register<ToolBar, bool>(nameof(IsOverflowOpen), defaultBindingMode: BindingMode.TwoWay,
            coerce: CoerceIsOverflowOpen);

    /// <summary>
    /// Gets or sets whether the "popup" for this control is currently open.
    /// </summary>
    public bool IsOverflowOpen
    {
        get => GetValue(IsOverflowOpenProperty);
        set => SetValue(IsOverflowOpenProperty, value);
    }

    private static bool CoerceIsOverflowOpen(AvaloniaObject obj, bool value)
    {
        if (value)
        {
            var tb = (ToolBar)obj;
            if (!tb.IsLoaded)
            {
                tb.RegisterToOpenOnLoad();
                return false;
            }
        }

        return value;
    }

    private void RegisterToOpenOnLoad()
    {
        Loaded += OpenOnLoad;
    }

    private void OpenOnLoad(object? sender, RoutedEventArgs e)
    {
        Dispatcher.UIThread.InvokeAsync(() => { CoerceValue(IsOverflowOpenProperty); }, DispatcherPriority.Input);
    }

    #endregion

    #region HasOverflowItems Property

    /// <summary>
    /// Defines the <see cref="HasOverflowItems"/> property.
    /// </summary>
    public static readonly StyledProperty<bool> HasOverflowItemsProperty =
        AvaloniaProperty.Register<ToolBar, bool>(nameof(HasOverflowItems));

    /// <summary>
    /// Gets or sets whether we have overflow items.
    /// </summary>
    public bool HasOverflowItems
    {
        get => GetValue(HasOverflowItemsProperty);
        set => SetValue(HasOverflowItemsProperty, value);
    }

    #endregion

    #region IsOverflowItem Property

    /// <summary>
    /// Defines the <see cref="IsOverflowItem"/> property.
    /// </summary>
    public static readonly StyledProperty<bool> IsOverflowItemProperty =
        AvaloniaProperty.Register<ToolBar, bool>(nameof(IsOverflowItem), inherits: true);

    /// <summary>
    /// Gets or sets whether the item overflow.
    /// </summary>
    public bool IsOverflowItem
    {
        get => GetValue(IsOverflowItemProperty);
        set => SetValue(IsOverflowItemProperty, value);
    }

    /// <summary>
    /// Sets whether the item overflow.
    /// </summary>
    /// <param name="control">Control</param>
    /// <param name="value">Value</param>
    public static void SetIsOverflowItem(Control control, bool value)
    {
        control.SetValue(IsOverflowItemProperty, value);
    }

    /// <summary>
    /// Gets whether the item overflow.
    /// </summary>
    /// <param name="control">Control</param>
    public static bool GetIsOverflowItem(Control control)
    {
        return control.GetValue(IsOverflowItemProperty);
    }

    #endregion

    #region OverflowMode Property

    /// <summary>
    /// Defines the <see cref="OverflowMode"/> property.
    /// </summary>
    public static readonly StyledProperty<OverflowMode> OverflowModeProperty =
        AvaloniaProperty.Register<ToolBar, OverflowMode>(nameof(OverflowMode),
            validate: IsValidOverflowMode);

    /// <summary>
    /// Gets or sets the overflow mode of the ToolBar.
    /// </summary>
    public OverflowMode OverflowMode
    {
        get => GetValue(OverflowModeProperty);
        set => SetValue(OverflowModeProperty, value);
    }

    private static void OnOverflowModeChanged(ToolBar toolBar, AvaloniaPropertyChangedEventArgs e)
    {
        // When OverflowMode changes on a child container of a ToolBar,
        // invalidate layout so that the child can be placed in the correct
        // location (in the main bar or the overflow menu).
        toolBar.InvalidateLayout();
    }

    private static bool IsValidOverflowMode(OverflowMode value)
    {
        return value == OverflowMode.AsNeeded
               || value == OverflowMode.Always
               || value == OverflowMode.Never;
    }

    #endregion

    #region MinVisibleItemsCount Property

    /// <summary>
    /// Defines the <see cref="MinVisibleItemsCount"/> property.
    /// </summary>
    public static readonly StyledProperty<uint> MinVisibleItemsCountProperty =
        AvaloniaProperty.Register<ToolBar, uint>(nameof(MinVisibleItemsCountProperty),
            defaultValue: 0);

    /// <summary>
    /// Get or set the count of items, which will be not overflowed while dragging the thumb.
    /// </summary>
    public uint MinVisibleItemsCount
    {
        get => GetValue(MinVisibleItemsCountProperty);
        set => SetValue(MinVisibleItemsCountProperty, value);
    }

    private static void OnMinVisibleItemsCountChanged(ToolBar toolBar, AvaloniaPropertyChangedEventArgs e)
    {
        // When OverflowMode changes on a child container of a ToolBar,
        // invalidate layout so that the child can be placed in the correct
        // location (in the main bar or the overflow menu).
        toolBar.InvalidateLayout();
    }

    #endregion

    static ToolBar()
    {
        OverflowModeProperty.Changed.AddClassHandler<ToolBar>(OnOverflowModeChanged);
        MinVisibleItemsCountProperty.Changed.AddClassHandler<ToolBar>(OnMinVisibleItemsCountChanged);

        IsTabStopProperty.OverrideMetadata<ToolBar>(new StyledPropertyMetadata<bool>(false));
        FocusableProperty.OverrideDefaultValue<ToolBar>(true);

        KeyboardNavigation.TabNavigationProperty.OverrideMetadata<ToolBar>(
            new StyledPropertyMetadata<KeyboardNavigationMode>(KeyboardNavigationMode.Cycle));

        Button.ClickEvent.AddClassHandler<ToolBar>((x, e) => x.OnClick(e));
    }
}