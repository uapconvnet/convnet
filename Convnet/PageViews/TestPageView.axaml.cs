using System;
using System.Data;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using Convnet.PageViewModels;
using ReactiveUI.Avalonia;

namespace Convnet.PageViews
{
    public partial class TestPageView : ReactiveUserControl<TestPageViewModel>
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public TestPageView()
        {
            InitializeComponent();

            var datagrid = this.FindControl<DataGrid>("Datagrid");
            if (datagrid != null)
            {
                datagrid.AutoGeneratingColumn += Datagrid_AutoGeneratingColumn;
                datagrid.DataContextChanged += Datagrid_DataContextChanged;
                datagrid.Loaded += Datagrid_Loaded;
                var tpvm = DataContext as TestPageViewModel;
                if (tpvm != null) 
                { 
                }
            }
        }

        private void Datagrid_Loaded(object? sender, Avalonia.Interactivity.RoutedEventArgs e) 
        {
            var datagrid = sender as DataGrid;
            if (datagrid != null)
            {
                
            }
        }

        private void Datagrid_AutoGeneratingColumn(object? sender, DataGridAutoGeneratingColumnEventArgs e)
        {
            e.Cancel = e.PropertyName == "DataView" || e.PropertyName == "Item" || e.PropertyName == "RowVersion" || e.PropertyName == "Row" || e.PropertyName == "IsNew" || e.PropertyName == "IsEdit";
        }
            
        private void Datagrid_DataContextChanged(object? sender, System.EventArgs e)
        {
            if (IsInitialized)
            {
                var datagrid = sender as DataGrid;
                var tpvm = DataContext as TestPageViewModel;
               
                if (datagrid != null && tpvm != null && tpvm.ConfusionDataView != null)
                {
                    Dispatcher.UIThread.Invoke(()=>
                    {
                        while (datagrid?.Columns.Count > 0) 
                            datagrid.Columns.RemoveAt(datagrid.Columns.Count - 1); 
                        
                        foreach (System.Data.DataColumn x in tpvm.ConfusionDataView.ToTable().Columns)
                            if (x.ColumnName == "RowHeader")
                                datagrid?.Columns.Add(new DataGridTextColumn { Header = "", Binding = new Avalonia.Data.Binding($"Row.ItemArray[{x.Ordinal}]") }); 
                            else
                                datagrid?.Columns.Add(new DataGridTextColumn { Header = x.ColumnName, Binding = new Avalonia.Data.Binding($"Row.ItemArray[{x.Ordinal}]") });
                    });
                }
            }
            else
                Initialized += delegate { Datagrid_DataContextChanged(sender, e); };
        }

        // private void UserControl_Loaded(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        // {
        //     var datagrid = this.FindControl<DataGrid>("Datagrid");
        //     var tpvm = DataContext as TestPageViewModel;

        //     if (datagrid != null && tpvm != null)
        //     {
        //         // datagrid?.ItemsSource = null;
        //         // datagrid?.ItemsSource = tpvm.ConfusionDataView;
        //         // datagrid?.Focus();
        //         // datagrid?.BringIntoView();
        //         // datagrid?.InvalidateVisual();
        //         // datagrid?.InvalidateArrange();
        //     }
        // }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
