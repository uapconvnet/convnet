namespace CustomMessageBox.Avalonia;

/// <summary>
/// Represents a button for a message box dialog with an associated result value.
/// </summary>
/// <typeparam name="TResult">The type of the result value. Must be a struct.</typeparam>
public struct MessageBoxButton<TResult> where TResult : struct
{
	/// <summary>
	/// Gets or sets the content of the button.
	/// </summary>
	public object Content { get; set; }
	/// <summary>
	/// Gets or sets the result value associated with this button.
	/// </summary>
	public TResult Result { get; set; }
	/// <summary>
	/// Gets or sets the class names for styling the button.
	/// </summary>
	public string[] ClassNames { get; set; }
	/// <summary>
	/// Gets or sets the special role assigned to this button.
	/// </summary>
	public SpecialButtonRole SpecialRole { get; set; }

	/// <summary>
	/// Initializes a new instance of the <see cref="MessageBoxButton{TResult}"/> struct with default class names.
	/// </summary>
	/// <param name="content">The content of the button.</param>
	/// <param name="result">The result value associated with this button.</param>
	/// <param name="specialRole">The special role assigned to this button.</param>
	public MessageBoxButton(object content, TResult result, SpecialButtonRole specialRole = SpecialButtonRole.None)
		: this(content, result, specialRole, string.Empty)
	{ }

	/// <summary>
	/// Initializes a new instance of the <see cref="MessageBoxButton{TResult}"/> struct with the specified class names.
	/// </summary>
	/// <param name="content">The content of the button.</param>
	/// <param name="result">The result value associated with this button.</param>
	/// <param name="specialRole">The special role assigned to this button.</param>
	/// <param name="classNames">The class names used to style the button.</param>
	public MessageBoxButton(object content, TResult result, SpecialButtonRole specialRole = SpecialButtonRole.None, params string[] classNames)
	{
		Content = content;
		Result = result;
		ClassNames = classNames;
		SpecialRole = specialRole;
	}
}
