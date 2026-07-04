namespace CustomMessageBox.Avalonia;

/// <summary>
/// Specifies the possible results of a message box interaction.
/// </summary>
public enum MessageBoxResult
{
	/// <summary>
	/// No result has been specified.
	/// </summary>
	None,

	/// <summary>
	/// The OK button was pressed.
	/// </summary>
	OK,

	/// <summary>
	/// The Cancel button was pressed.
	/// </summary>
	Cancel,

	/// <summary>
	/// The Abort button was pressed.
	/// </summary>
	Abort,

	/// <summary>
	/// The Retry button was pressed.
	/// </summary>
	Retry,

	/// <summary>
	/// The Ignore button was pressed.
	/// </summary>
	Ignore,

	/// <summary>
	/// The Yes button was pressed.
	/// </summary>
	Yes,

	/// <summary>
	/// The No button was pressed.
	/// </summary>
	No,

	/// <summary>
	/// The Try Again button was pressed.
	/// </summary>
	TryAgain,

	/// <summary>
	/// The Continue button was pressed.
	/// </summary>
	Continue
}
