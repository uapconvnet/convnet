using Avalonia;
using Avalonia.Controls.Metadata;
using Avalonia.Controls.Primitives;
using Avalonia.Input;

namespace ToolBarControls.Avalonia;

/// <summary>
/// Thumb for <see cref="ToolBar"/> control.
/// </summary>
[PseudoClasses(":pressed")]
public partial class ToolBarThumb : Thumb
{
    private Point? _lastPoint;

    /// <inheritdoc />
    protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e)
    {
        if (_lastPoint.HasValue)
        {
            var ev = new ToolBarVectorEventArgs(null)
            {
                RoutedEvent = DragCompletedEvent,
                Vector = _lastPoint.Value
            };

            _lastPoint = null;

            RaiseEvent(ev);
        }

        PseudoClasses.Remove(":pressed");

        base.OnPointerCaptureLost(e);
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        if (_lastPoint.HasValue)
        {
            var ev = new ToolBarVectorEventArgs(e)
            {
                RoutedEvent = DragDeltaEvent,
                Vector = e.GetPosition(this) - _lastPoint.Value
            };

            RaiseEvent(ev);
        }
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        e.Handled = true;
        _lastPoint = e.GetPosition(this);

        var ev = new ToolBarVectorEventArgs(e)
        {
            RoutedEvent = DragStartedEvent,
            Vector = (Vector)_lastPoint
        };

        PseudoClasses.Add(":pressed");

        e.PreventGestureRecognition();

        RaiseEvent(ev);
    }

    /// <inheritdoc />
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        if (_lastPoint.HasValue)
        {
            e.Handled = true;
            _lastPoint = null;

            var ev = new ToolBarVectorEventArgs(e)
            {
                RoutedEvent = DragCompletedEvent,
                Vector = e.GetPosition(this)
            };

            RaiseEvent(ev);
        }

        PseudoClasses.Remove(":pressed");
    }
}