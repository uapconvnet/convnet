using System;
using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Metadata;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.LogicalTree;
using Avalonia.VisualTree;

namespace ToolBarControls.Avalonia;

/// <summary>
/// ToolBar provides an overflow mechanism which places any items that doesnt fit naturally
/// fit within a size-constrained ToolBar into a special overflow area.
/// Also, ToolBars have a tight relationship with the related ToolBarTray control.
/// </summary>
[TemplatePart(ElementToolBarPanel, typeof(ToolBarPanel), IsRequired = true)]
[TemplatePart(ElementToolBarOverflowPanel, typeof(ToolBarOverflowPanel), IsRequired = true)]
[TemplatePart(ElementOverflowButton, typeof(ToggleButton), IsRequired = true)]
[TemplatePart(ElementToolBarPopup, typeof(Popup), IsRequired = true)]
public partial class ToolBar : HeaderedItemsControl
{
    /// <inheritdoc />
    protected override Type StyleKeyOverride => typeof(ToolBar);

    private const string PcDropdownOpen = ":dropdownopen";

    private const string ElementToolBarOverflowPanel = "PART_ToolBarOverflowPanel";
    private const string ElementToolBarPanel = "PART_ToolBarPanel";
    private const string ElementOverflowButton = "PART_OverflowButton";
    private const string ElementToolBarPopup = "PART_OverflowPopup";

    private ToggleButton? _overflowButton;
    private Popup? _popup;

    /// <inheritdoc />
    public ToolBar()
    {
        Items.CollectionChanged += OnItemsChanged;
    }

    /// <summary>
    /// Gets reference to ToolBar's ToolBarPanel element.
    /// </summary>
    internal ToolBarPanel? ToolBarPanel { get; private set; }

    /// <summary>
    /// Gets reference to ToolBar's ToolBarOverflowPanel element.
    /// </summary>
    internal ToolBarOverflowPanel? ToolBarOverflowPanel { get; private set; }

    internal double MinLength { get; private set; }
    internal double MaxLength { get; private set; }
    private ToolBarTray? ToolBarTray => Parent as ToolBarTray;

    /// <inheritdoc />
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        if (change.Property == BandProperty || change.Property == BandIndexProperty)
        {
            if (Parent is not Layoutable visualParent)
                return;

            visualParent.InvalidateMeasure();
        }
        else if (change.Property == IsOverflowOpenProperty)
        {
            PseudoClasses.Set(PcDropdownOpen, change.GetNewValue<bool>());
        }

        base.OnPropertyChanged(change);
    }

    /// <inheritdoc />
    protected override void OnApplyTemplate(TemplateAppliedEventArgs e)
    {
        base.OnApplyTemplate(e);

        ToolBarPanel = e.NameScope.Find<ToolBarPanel>(ElementToolBarPanel);
        ToolBarOverflowPanel = e.NameScope.Find<ToolBarOverflowPanel>(ElementToolBarOverflowPanel);
        _overflowButton = e.NameScope.Find<ToggleButton>(ElementOverflowButton);
        _popup = e.NameScope.Find<Popup>(ElementToolBarPopup);
    }

    /// <inheritdoc />
    protected override void OnTemplateChanged(AvaloniaPropertyChangedEventArgs e)
    {
        // Invalidate template references
        ToolBarPanel = null;
        ToolBarOverflowPanel = null;

        base.OnTemplateChanged(e);
    }

    /// <inheritdoc />
    protected override Size MeasureOverride(Size constraint)
    {
        // Perform a normal layout
        var desiredSize = base.MeasureOverride(constraint);

        // MinLength and MaxLength are used by ToolBarTray to determine
        // its layout. ToolBarPanel will calculate its version of these values.
        // ToolBar needs to add on the space used up by elements around the ToolBarPanel.
        //
        // Note: This calculation is not 100% accurate. If a scale transform is applied
        // within the template of the ToolBar (between the ToolBar and the ToolBarPanel),
        // then the coordinate spaces will not match and the values will be wrong.
        //
        // Note: If a ToolBarPanel is not contained within the ToolBar's template,
        // then these values will always be zero, and ToolBarTray will not layout correctly.
        //
        var toolBarPanel = ToolBarPanel;
        if (toolBarPanel != null)
        {
            // Calculate the extra length from the extra space allocated between the ToolBar and the ToolBarPanel.
            double extraLength;
            Thickness margin = toolBarPanel.Margin;
            if (toolBarPanel.Orientation == Orientation.Horizontal)
            {
                extraLength = Math.Max(0.0,
                    desiredSize.Width - toolBarPanel.DesiredSize.Width + margin.Left + margin.Right);
            }
            else
            {
                extraLength = Math.Max(0.0,
                    desiredSize.Height - toolBarPanel.DesiredSize.Height + margin.Top + margin.Bottom);
            }

            // Add the calculated extra length to the lengths provided by ToolBarPanel
            MinLength = toolBarPanel.MinLength + extraLength;
            MaxLength = toolBarPanel.MaxLength + extraLength;
        }

        return desiredSize;
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (e is { Handled: false, Source: Visual source })
        {
            if (_popup?.IsInsidePopup(source) == true)
            {
                e.Handled = true;
                return;
            }
        }

        if (IsOverflowOpen)
        {
            SetCurrentValue(IsOverflowOpenProperty, false);
            e.Handled = true;
        }
    }

    /// <inheritdoc />
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        if (e is { Handled: false, Source: Visual source })
        {
            // Need for not to close SelectingItemsControl Popup
            if (_popup?.IsInsidePopup(source) == true &&
                source.FindAncestorOfType<SelectingItemsControl>() == null &&
                source.FindAncestorOfType<AutoCompleteBox>() == null &&
                source.FindAncestorOfType<AutoCompleteBox>() == null)
            {
                SetCurrentValue(IsOverflowOpenProperty, false);
                e.Handled = true;
            }
        }

        base.OnPointerReleased(e);
    }

    internal void AddLogicalChild(Control c)
    {
        if (!LogicalChildren.Contains(c))
            LogicalChildren.Add(c);
    }

    internal void RemoveLogicalChild(Control c) => LogicalChildren.Remove(c);

    private void OnItemsChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        // When items change, invalidate layout so that the decision
        // regarding which items are in the overflow menu can be re-done.
        InvalidateLayout();
    }

    private void InvalidateLayout()
    {
        // Reset the calculated min and max size
        MinLength = 0.0;
        MaxLength = 0.0;

        // Min and max sizes are calculated in ToolBar.MeasureOverride
        InvalidateMeasure();

        var toolBarPanel = ToolBarPanel;
        if (toolBarPanel != null)
        {
            // Whether elements are in the overflow or not is decided
            // in ToolBarPanel.MeasureOverride.
            toolBarPanel.InvalidateMeasure();
        }
    }

    private void OnClick(RoutedEventArgs e)
    {
        if (Equals(e.Source, _overflowButton))
            return;

        if (IsOverflowOpen && e.Source is Button b &&
            Equals(b.FindLogicalAncestorOfType<ToolBar>(), this))
        {
            SetCurrentValue(IsOverflowOpenProperty, false);
        }
    }
}