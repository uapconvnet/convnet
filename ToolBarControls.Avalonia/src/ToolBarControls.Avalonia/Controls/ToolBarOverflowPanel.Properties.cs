using Avalonia;
using ToolBarControls.Avalonia.Utils;

namespace ToolBarControls.Avalonia;

public partial class ToolBarOverflowPanel
{
    #region WrapWidth Property

    /// <summary>
    /// Defines the <see cref="WrapWidth"/> property.
    /// </summary>
    public static readonly StyledProperty<double> WrapWidthProperty =
        AvaloniaProperty.Register<ToolBarOverflowPanel, double>(nameof(WrapWidth), double.NaN,
            validate: IsWrapWidthValid);

    /// <summary>
    /// Get or set wrap width.
    /// </summary>
    public double WrapWidth
    {
        get => GetValue(WrapWidthProperty);
        set => SetValue(WrapWidthProperty, value);
    }

    static bool IsWrapWidthValid(double v)
    {
        return double.IsNaN(v) || DoubleUtil.GreaterThanOrClose(v, 0d) && !double.IsPositiveInfinity(v);
    }

    #endregion

    static ToolBarOverflowPanel()
    {
        AffectsMeasure<ToolBarOverflowPanel>(WrapWidthProperty);
    }
}