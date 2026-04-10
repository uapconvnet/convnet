using Avalonia.Input;

namespace ToolBarControls.Avalonia;

/// <summary>
/// Custom VectorEventArgs for <see cref="ToolBarThumb"/> control.
/// </summary>
public class ToolBarVectorEventArgs : VectorEventArgs
{
    /// <inheritdoc />
    public ToolBarVectorEventArgs(PointerEventArgs? pointerEventArgs)
    {
        PointerEventArgs = pointerEventArgs;
    }

    /// <summary>
    /// PointerEventArgs
    /// </summary>
    public PointerEventArgs? PointerEventArgs { get; }
}