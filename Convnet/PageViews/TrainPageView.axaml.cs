using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using Avalonia.Threading;
using Convnet.PageViewModels;
using CustomMessageBox.Avalonia;
using Interop;
using System.Collections.Generic;

namespace Convnet.PageViews
{
    public partial class TrainPageView : ReactiveUserControl<TrainPageViewModel>
    {
        private bool zoomout = false;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public TrainPageView()
        {
            InitializeComponent();

            var datagrid = this.FindControl<DataGrid>("ListViewTrainingResult");
            if (datagrid != null)
                datagrid.DataContextChanged += Datagrid_DataContextChanged;
        }

        private void Datagrid_DataContextChanged(object? sender, System.EventArgs e)
        {
            var datagrid = sender as DataGrid;
            if (datagrid != null)
            {
                var tpvm = datagrid.DataContext as TrainPageViewModel;
                if (tpvm != null && tpvm.TrainingLog != null)
                {
                    Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        if (tpvm.SelectedItems != null && datagrid.SelectionMode == DataGridSelectionMode.Extended)
                        {
                            tpvm.IsUpdating = true;

                            datagrid.SelectedItems.Clear();
                            foreach (var item in tpvm.SelectedItems)
                                foreach (var row in tpvm.TrainingLog)
                                    if (item.Equals(row))
                                        datagrid.SelectedItems.Add(row);
                            
                            if (datagrid.SelectedItems.Count == 0)
                                tpvm.SelectedItems.Clear();

                            tpvm.IsUpdating = false;
                        }

                        if (tpvm.SelectedIndex >= 0 && tpvm.SelectedIndex < tpvm.TrainingLog.Count)
                        {
                            if (datagrid.SelectionMode == DataGridSelectionMode.Single)
                                datagrid.SelectedIndex = tpvm.SelectedIndex;

                            datagrid.ScrollIntoView(tpvm.TrainingLog[tpvm.SelectedIndex], null);
                        }
                        else
                            datagrid.ScrollIntoView(tpvm.TrainingLog[0], null);

                        datagrid.Focus();

                    }, DispatcherPriority.ContextIdle);
                }
            }
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private void ZoomOut_PointerPressed(object sender, PointerPressedEventArgs e)
        {
            var point = e.GetCurrentPoint(sender as Control);

            if (point.Properties.IsLeftButtonPressed && e.ClickCount == 2)
            {
                var gridMain = this.FindControl<Grid>("GridMain");
                var borderSnapShot = this.FindControl<Border>("BorderSnapShot");
                var borderTrainingPlot = this.FindControl<Border>("BorderTrainingPlot");
                var borderWeightsMinMax = this.FindControl<Border>("BorderWeightsMinMax");

                if (gridMain != null && borderSnapShot != null && borderTrainingPlot != null && borderWeightsMinMax != null)
                {
                    if (!zoomout)
                    {
                        borderSnapShot.SetValue(Grid.ColumnProperty, 0);
                        borderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
                        borderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
                        borderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

                        gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
                        gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
                        gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
                        zoomout = true;
                    }
                    else
                    {
                        borderSnapShot.SetValue(Grid.ColumnProperty, 1);
                        borderSnapShot.SetValue(Grid.ColumnSpanProperty, 1);
                        borderTrainingPlot.SetValue(Grid.ColumnProperty, 1);
                        borderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 1);

                        gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Auto);
                        gridMain.RowDefinitions[1].Height = new GridLength(26.0, GridUnitType.Pixel);
                        gridMain.RowDefinitions[2].Height = new GridLength(1.0, GridUnitType.Star);

                        borderSnapShot.MaxHeight = 0.0;
                        borderSnapShot.UpdateLayout();

                        Binding binding = new Binding
                        {
                            Path = "Bounds.Height",
                            Source = borderWeightsMinMax
                        };
                        borderSnapShot.Bind(Border.MaxHeightProperty, binding);

                        borderTrainingPlot.MaxHeight = 0.0;
                        borderTrainingPlot.UpdateLayout();

                        binding = new Binding
                        {
                            Path = "Bounds.Height",
                            Source = borderWeightsMinMax
                        };
                        borderTrainingPlot.Bind(Border.MaxHeightProperty, binding);
                        zoomout = false;
                    }

                    e.Handled = true;
                }
            }
        }

        private async void DataGrid_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Delete)
            {
                var datagrid = this.FindControl<DataGrid>("ListViewTrainingResult");
                if (datagrid != null)
                {
                    var tpvm = datagrid.DataContext as TrainPageViewModel;
                    
                    if (tpvm != null && tpvm.TrainingLog != null && datagrid.SelectedItems.Count > 0)
                    {
                        var result = await Dispatcher.UIThread.InvokeAsync(() => MessageBox.Show("Do you really want to delete the selected row(s)?", "Delete row(s)", MessageBoxButtons.YesNo, MessageBoxIcon.None, MessageBoxDefaultButton.Button2));
                        if (result == MessageBoxResult.Yes)
                        {
                            List<DNNTrainingResult> items = new List<DNNTrainingResult>();

                            foreach (var item in datagrid.SelectedItems)
                                if (item is DNNTrainingResult row)
                                    if (row != null)
                                        items.Add(row);

                            foreach (var item in items)
                                tpvm.TrainingLog?.Remove(item);

                            datagrid.SelectedItems.Clear();
                        }
                    }
                }
            }
        }
    }
}
