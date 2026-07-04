namespace CustomMessageBox.Avalonia;

/// <summary>
/// Specifies which button is the default for a message box.
/// </summary>
public enum MessageBoxDefaultButton
{
	/// <summary>
	/// No default button is selected.
	/// </summary>
	None = -1,

	/// <summary>
	/// The first button is the default.
	/// </summary>
	Button1 = 0,

	/// <summary>
	/// The second button is the default.
	/// </summary>
	Button2 = 1,

	/// <summary>
	/// The third button is the default.
	/// </summary>
	Button3 = 2
}
