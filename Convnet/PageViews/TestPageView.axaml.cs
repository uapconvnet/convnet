using System;
using System.Data;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Convnet.PageViewModels;
using ReactiveUI.Avalonia;

namespace Convnet.PageViews
{
    public partial class TestPageView : ReactiveUserControl<TestPageViewModel>
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public TestPageView()
        {
            //string[] names = this.GetType().Assembly.GetManifestResourceNames();
            //string[] anames = Assembly.GetExecutingAssembly().GetManifestResourceNames();

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
            e.Cancel = (e.PropertyName == "DataView" || e.PropertyName == "Item" || e.PropertyName == "RowVersion" || e.PropertyName == "Row" || e.PropertyName == "IsNew" || e.PropertyName == "IsEdit");
        }
            
        private void Datagrid_DataContextChanged(object? sender, System.EventArgs e)
        {
            var datagrid = sender as DataGrid;
            if (datagrid != null)
            {
                var tpvm = DataContext as TestPageViewModel;
                if (tpvm != null && tpvm.ConfusionDataTable != null)
                {
                    while (datagrid.Columns.Count > 0) { datagrid.Columns.RemoveAt(datagrid.Columns.Count - 1); }

                    datagrid.ItemsSource = tpvm.ConfusionDataTable.DefaultView;

                    foreach (System.Data.DataColumn x in tpvm.ConfusionDataTable.Columns)
                        datagrid.Columns.Add(new DataGridTextColumn { Header = x.ColumnName, Binding = new Avalonia.Data.Binding($"Row.ItemArray[{x.Ordinal}]") });
                }
            }
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
