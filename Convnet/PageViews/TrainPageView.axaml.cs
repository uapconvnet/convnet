using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Markup.Xaml;

namespace Convnet.PageViews
{
    public partial class TrainPageView : UserControl
    {
        private bool Zoomout = false;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
        public TrainPageView()
        {
            //string[] names = this.GetType().Assembly.GetManifestResourceNames();
            //string[] anames = Assembly.GetExecutingAssembly().GetManifestResourceNames();
           
            InitializeComponent();


            //var gr = this.FindControl<Grid>("grid");
        }
      
        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private void TrainingPlot_PointerPressed(object sender, PointerPressedEventArgs e)
        {
            var point = e.GetCurrentPoint(sender as Control);

            if (point.Properties.IsLeftButtonPressed && e.ClickCount == 2)
            {
                if (!Zoomout)
                {
                    BorderSnapShot.SetValue(Grid.ColumnProperty, 0);
                    BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
                    BorderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
                    BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

                    gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
                    gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
                    gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
                    Zoomout = true;
                }
                else
                {
                    BorderSnapShot.SetValue(Grid.ColumnProperty, 1);
                    BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 1);
                    BorderTrainingPlot.SetValue(Grid.ColumnProperty, 1);
                    BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 1);

                    gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Auto);
                    gridMain.RowDefinitions[1].Height = new GridLength(20.0, GridUnitType.Pixel);
                    gridMain.RowDefinitions[2].Height = new GridLength(1.0, GridUnitType.Star);

                    BorderSnapShot.MaxHeight = 0.0;
                    BorderSnapShot.UpdateLayout();

                    Binding binding = new Binding
                    {
                        Path = "ActualHeight",
                        Source = BorderWeightsMinMax
                    };
                    BorderSnapShot.Bind(Border.MaxHeightProperty, binding);

                    BorderTrainingPlot.MaxHeight = 0.0;
                    BorderTrainingPlot.UpdateLayout();

                    binding = new Binding
                    {
                        Path = "ActualHeight",
                        Source = BorderWeightsMinMax
                    };
                    BorderTrainingPlot.Bind(Border.MaxHeightProperty, binding);
                    Zoomout = false;
                }

                e.Handled = true;
            }
        }

        private void SnapShot_PointerPressed(object sender, PointerPressedEventArgs e)
        {
            var point = e.GetCurrentPoint(sender as Control);

            if (point.Properties.IsLeftButtonPressed && e.ClickCount == 2)
            {
                if (!Zoomout)
                {
                    BorderSnapShot.SetValue(Grid.ColumnProperty, 0);
                    BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
                    BorderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
                    BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

                    gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
                    gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
                    gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
                    Zoomout = true;
                }
                else
                {
                    BorderSnapShot.SetValue(Grid.ColumnProperty, 1);
                    BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 1);
                    BorderTrainingPlot.SetValue(Grid.ColumnProperty, 1);
                    BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 1);

                    gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Auto);
                    gridMain.RowDefinitions[1].Height = new GridLength(20.0, GridUnitType.Pixel);
                    gridMain.RowDefinitions[2].Height = new GridLength(1.0, GridUnitType.Star);

                    BorderSnapShot.MaxHeight = 0.0;
                    BorderSnapShot.UpdateLayout();

                    Binding binding = new Binding
                    {
                        Path = "ActualHeight",
                        Source = BorderWeightsMinMax
                    };
                    BorderSnapShot.Bind(Border.MaxHeightProperty, binding);

                    BorderTrainingPlot.MaxHeight = 0.0;
                    BorderTrainingPlot.UpdateLayout();

                    binding = new Binding
                    {
                        Path = "ActualHeight",
                        Source = BorderWeightsMinMax
                    };
                    BorderTrainingPlot.Bind(Border.MaxHeightProperty, binding);

                    Zoomout = false;
                }

                e.Handled = true;
            }
        }
    }
}
