using System;
using Avalonia.Interactivity;

namespace ToolBarControls.Avalonia;

public partial class ToolBarThumb
{
    #region DragStarted Event

    /// <summary>
    /// Defines the <see cref="DragStarted"/> event.
    /// </summary>
    public new static readonly RoutedEvent<ToolBarVectorEventArgs> DragStartedEvent =
        RoutedEvent.Register<ToolBarThumb, ToolBarVectorEventArgs>(nameof(DragStarted), RoutingStrategies.Bubble);

    /// <summary>
    /// Event for handle start dragging thumb state.
    /// </summary>
    public new event EventHandler<ToolBarVectorEventArgs> DragStarted
    {
        add => AddHandler(DragStartedEvent, value);
        remove => RemoveHandler(DragStartedEvent, value);
    }

    #endregion

    #region DragDelta Event

    /// <summary>
    /// Defines the <see cref="DragDelta"/> event.
    /// </summary>
    public new static readonly RoutedEvent<ToolBarVectorEventArgs> DragDeltaEvent =
        RoutedEvent.Register<ToolBarThumb, ToolBarVectorEventArgs>(nameof(DragDelta), RoutingStrategies.Bubble);

    /// <summary>
    /// Event for handle dragging delta thumb state.
    /// </summary>
    public new event EventHandler<ToolBarVectorEventArgs> DragDelta
    {
        add => AddHandler(DragDeltaEvent, value);
        remove => RemoveHandler(DragDeltaEvent, value);
    }

    #endregion

    #region DragCompleted Event

    /// <summary>
    /// Defines the <see cref="DragCompleted"/> event.
    /// </summary>
    public new static readonly RoutedEvent<ToolBarVectorEventArgs> DragCompletedEvent =
        RoutedEvent.Register<ToolBarThumb, ToolBarVectorEventArgs>(nameof(DragCompleted), RoutingStrategies.Bubble);

    /// <summary>
    /// Event for handle complete dragging thumb state.
    /// </summary>
    public new event EventHandler<ToolBarVectorEventArgs> DragCompleted
    {
        add => AddHandler(DragCompletedEvent, value);
        remove => RemoveHandler(DragCompletedEvent, value);
    }

    #endregion

    static ToolBarThumb()
    {
        DragStartedEvent.AddClassHandler<ToolBarThumb>((x, e) => x.OnDragStarted(e), RoutingStrategies.Bubble);
        DragDeltaEvent.AddClassHandler<ToolBarThumb>((x, e) => x.OnDragDelta(e), RoutingStrategies.Bubble);
        DragCompletedEvent.AddClassHandler<ToolBarThumb>((x, e) => x.OnDragCompleted(e), RoutingStrategies.Bubble);
    }

    /// <summary>
    /// Handle for <see cref="DragStartedEvent"/> event.
    /// </summary>
    protected virtual void OnDragStarted(ToolBarVectorEventArgs e)
    {
    }

    /// <summary>
    /// Handle for <see cref="DragDeltaEvent"/> event.
    /// </summary>
    protected virtual void OnDragDelta(ToolBarVectorEventArgs e)
    {
    }

    /// <summary>
    /// Handle for <see cref="DragCompletedEvent"/> event.
    /// </summary>
    protected virtual void OnDragCompleted(ToolBarVectorEventArgs e)
    {
    }
}