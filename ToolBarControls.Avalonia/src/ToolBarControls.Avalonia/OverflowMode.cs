namespace ToolBarControls.Avalonia;

/// <summary>
/// Defines how we place the toolbar items
/// </summary>
public enum OverflowMode
{
    /// <summary>
    /// specifies that the item moves between the main and the overflow panels as space permits
    /// </summary>
    AsNeeded,

    /// <summary>
    /// specifies that the item is permanently placed in the overflow panel
    /// </summary>
    Always,

    /// <summary>
    /// specifies that the item is never allowed to overflow
    /// </summary>
    Never

    // NOTE: if you add or remove any values in this enum, be sure to update ToolBar.IsValidOverflowMode()
}