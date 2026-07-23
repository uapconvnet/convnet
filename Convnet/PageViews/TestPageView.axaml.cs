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

                if (datagrid != null && tpvm != null && tpvm.ConfusionDataTable != null)
                {
                    Dispatcher.Invoke(()=>
                    {
                        while (datagrid.Columns.Count > 0) { datagrid.Columns.RemoveAt(datagrid.Columns.Count - 1); }

                        datagrid.ItemsSource = tpvm.ConfusionDataTable.DefaultView;

                        foreach (System.Data.DataColumn x in tpvm.ConfusionDataTable.Columns)
                        if (x.ColumnName == "RowHeader")
                            datagrid.Columns.Add(new DataGridTextColumn { Header = "", Binding = new Avalonia.Data.Binding($"Row.ItemArray[{x.Ordinal}]") }); 
                        else
                            datagrid.Columns.Add(new DataGridTextColumn { Header = x.ColumnName, Binding = new Avalonia.Data.Binding($"Row.ItemArray[{x.Ordinal}]") });        
                    }, DispatcherPriority.ContextIdle);
                }
            }
            else
                Initialized += delegate { Datagrid_DataContextChanged(sender, e); };
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
