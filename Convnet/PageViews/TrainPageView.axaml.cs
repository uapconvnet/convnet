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
            InitializeComponent();
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
                var borderSnapShot = this.FindControl<Border>("BorderSnapShot");
                var borderTrainingPlot = this.FindControl<Border>("BorderTrainingPlot");
                var gridMain = this.FindControl<Grid>("GridMain");
                var borderWeightsMinMax = this.FindControl<Border>("BorderWeightsMinMax");

                if (borderSnapShot != null && borderTrainingPlot != null && gridMain != null && borderWeightsMinMax != null)
                {
                    if (!Zoomout)
                    {
                        borderSnapShot.SetValue(Grid.ColumnProperty, 0);
                        borderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
                        borderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
                        borderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

                        gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
                        gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
                        gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
                        Zoomout = true;
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
                            Path = "ActualHeight",
                            Source = borderWeightsMinMax
                        };
                        borderSnapShot.Bind(Border.MaxHeightProperty, binding);

                        borderTrainingPlot.MaxHeight = 0.0;
                        borderTrainingPlot.UpdateLayout();

                        binding = new Binding
                        {
                            Path = "ActualHeight",
                            Source = borderWeightsMinMax
                        };
                        borderTrainingPlot.Bind(Border.MaxHeightProperty, binding);
                        Zoomout = false;
                    }

                    e.Handled = true;
                }
            }
        }
        
        private void SnapShot_PointerPressed(object sender, PointerPressedEventArgs e)
        {
            var point = e.GetCurrentPoint(sender as Control);

            if (point.Properties.IsLeftButtonPressed && e.ClickCount == 2)
            {
                var borderSnapShot = this.FindControl<Border>("BorderSnapShot");
                var borderTrainingPlot = this.FindControl<Border>("BorderTrainingPlot");
                var gridMain = this.FindControl<Grid>("GridMain");
                var borderWeightsMinMax = this.FindControl<Border>("BorderWeightsMinMax");

                if (borderSnapShot != null && borderTrainingPlot != null && gridMain != null && borderWeightsMinMax != null)
                {
                    if (!Zoomout)
                    {
                        borderSnapShot.SetValue(Grid.ColumnProperty, 0);
                        borderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
                        borderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
                        borderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

                        gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
                        gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
                        gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
                        Zoomout = true;
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
                            Path = "ActualHeight",
                            Source = borderWeightsMinMax
                        };
                        borderSnapShot.Bind(Border.MaxHeightProperty, binding);

                        borderTrainingPlot.MaxHeight = 0.0;
                        borderTrainingPlot.UpdateLayout();

                        binding = new Binding
                        {
                            Path = "ActualHeight",
                            Source = borderWeightsMinMax
                        };
                        borderTrainingPlot.Bind(Border.MaxHeightProperty, binding);
                        Zoomout = false;
                    }

                    e.Handled = true;
                }
            }
        }
    }
}
