namespace CustomMessageBox.Avalonia;

/// <summary>
/// Specifies special roles that a button can have within a message box.
/// </summary>
public enum SpecialButtonRole
{
	/// <summary>
	/// No special role.
	/// </summary>
	None,

	/// <summary>
	/// Indicates the button is the default action (activated when Enter is pressed).
	/// </summary>
	IsDefault,

	/// <summary>
	/// Indicates the button is the cancel action (activated when Escape is pressed).
	/// </summary>
	IsCancel
}
