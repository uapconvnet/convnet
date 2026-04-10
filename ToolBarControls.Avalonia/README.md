# ToolBarControls.Avalonia

[![NuGet](https://img.shields.io/nuget/v/Tulesha.ToolBarControls.Avalonia)](https://www.nuget.org/packages/Tulesha.ToolBarControls.Avalonia) [![downloads](https://img.shields.io/nuget/dt/Tulesha.ToolBarControls.Avalonia)](https://www.nuget.org/packages/Tulesha.ToolBarControls.Avalonia)

## Description

This package provides two custom controls for Avalonia UI: `ToolBar` and `ToolBarTray`. These controls enable the
creation
of flexible, dockable toolbars with overflow functionality similar to traditional desktop applications. The
`ToolBarTray`
acts as a container that can hold multiple `ToolBar` instances, allowing them to be organized in bands and supporting
both
horizontal and vertical orientations. The controls include features like drag-and-drop repositioning, overflow menus for
space-constrained scenarios, and keyboard navigation support.

## API

**ToolBar Properties**

| PROPERTY NAME        | TYPE           | DESCRIPTION                                                                                                        |
|----------------------|----------------|--------------------------------------------------------------------------------------------------------------------|
| Orientation          | `Orientation`  | Gets or sets the orientation of the ToolBar. This property is inherited and coerced by the parent ToolBarTray.     |
| Band                 | `int`          | Gets or sets the band number where ToolBar should be located within the ToolBarTray.                               |
| BandIndex            | `int`          | Gets or sets the band index number where ToolBar should be located within the band of ToolBarTray.                 |
| IsOverflowOpen       | `bool`         | Gets or sets whether the overflow popup for this control is currently open. Supports two-way binding.              |
| HasOverflowItems     | `bool`         | Gets or sets whether the ToolBar has overflow items.                                                               |
| IsOverflowItem       | `bool`         | Gets or sets whether the item should overflow. This is an attached property that can be applied to child controls. |
| OverflowMode         | `OverflowMode` | Gets or sets the overflow mode of the ToolBar. Valid values are AsNeeded, Always, and Never.                       |
| MinVisibleItemsCount | `uint`         | Gets or sets the count of items that will not be overflowed while resizing the control.                            |

**ToolBarTray Properties**

| PROPERTY NAME | TYPE                  | DESCRIPTION                                                                                                            |
|---------------|-----------------------|------------------------------------------------------------------------------------------------------------------------|
| Background    | `IBrush?`             | Gets or sets the background brush of the ToolBarTray.                                                                  |
| Orientation   | `Orientation`         | Gets or sets the orientation of the ToolBarTray (Horizontal or Vertical). Default is Horizontal.                       |
| IsLocked      | `bool`                | Gets or sets whether the ToolBarTray is locked, preventing drag-and-drop repositioning of contained ToolBars.          |
| ToolBars      | `Collection<ToolBar>` | Gets the collection of ToolBar controls contained within the ToolBarTray. This is the content property of the control. |

## Demos

**ToolBar Demo**

![](https://github.com/Tulesha/ToolBarControls.Avalonia/blob/main/workflows/ToolBarSample.gif)

**ToolBarTray Demo**

![](https://github.com/Tulesha/ToolBarControls.Avalonia/blob/main/workflows/ToolBarTraySample.gif)
