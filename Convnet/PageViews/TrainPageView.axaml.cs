using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace Convnet.PageViews
{
    private bool Zoomout = false;

    public partial class TrainPageView : UserControl
    {
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
    }

    //  private void SnapShot_MouseLeftButtonDown(object sender, PointerEventArgs args)
    //  {
    //         var point = args.GetCurrentPoint(sender as Control);
    //         var x = point.Position.X;
    //         var y = point.Position.Y;
    //         var msg = $"Pointer press at {x}, {y} relative to sender.";
    //         if (point.Properties.IsLeftButtonPressed)
    //         {
    //             msg += " Left button pressed.";
    //         }

    //         if (e.ClickCount == 2)
    //         {
    //             if (!Zoomout)
    //             {
    //                 BorderSnapShot.SetValue(Grid.ColumnProperty, 0);
    //                 BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
    //                 BorderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
    //                 BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

    //                 gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
    //                 gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
    //                 gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
    //                 Zoomout = true;
    //             }
    //             else
    //             {
    //                 BorderSnapShot.SetValue(Grid.ColumnProperty, 1);
    //                 BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 1);
    //                 BorderTrainingPlot.SetValue(Grid.ColumnProperty, 1);
    //                 BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 1);

    //                 gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Auto);
    //                 gridMain.RowDefinitions[1].Height = new GridLength(20.0, GridUnitType.Pixel);
    //                 gridMain.RowDefinitions[2].Height = new GridLength(1.0, GridUnitType.Star);

    //                 BorderSnapShot.MaxHeight = 0.0;
    //                 BorderSnapShot.UpdateLayout();

    //                 Binding binding = new Binding
    //                 {
    //                     Path = new PropertyPath("ActualHeight"),
    //                     Source = BorderWeightsMinMax
    //                 };
    //                 BorderSnapShot.SetBinding(Border.MaxHeightProperty, binding);

    //                 BorderTrainingPlot.MaxHeight = 0.0;
    //                 BorderTrainingPlot.UpdateLayout();

    //                 binding = new Binding
    //                 {
    //                     Path = new PropertyPath("ActualHeight"),
    //                     Source = BorderWeightsMinMax
    //                 };
    //                 BorderTrainingPlot.SetBinding(Border.MaxHeightProperty, binding);

    //                 Zoomout = false;
    //             }

    //             e.Handled = true;
    //         }
    //     } 

        // private void TrainingPlot_MouseLeftButtonDown(object sender, PointerEventArgs args)
        // {
        //     var point = args.GetCurrentPoint(sender as Control);
           
        //     if (point.Properties.IsLeftButtonPressed && args.ClickCount == 2)
        //     {
        //          if (!Zoomout)
        //          {
        //     //         BorderSnapShot.SetValue(Grid.ColumnProperty, 0);
        //     //         BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 2);
        //     //         BorderTrainingPlot.SetValue(Grid.ColumnProperty, 0);
        //     //         BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 2);

        //     //         gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Star);
        //     //         gridMain.RowDefinitions[1].Height = new GridLength(0.0, GridUnitType.Pixel);
        //     //         gridMain.RowDefinitions[2].Height = new GridLength(0.0, GridUnitType.Pixel);
        //     //         Zoomout = true;
        //          }
        //          else
        //          {
        //     //         BorderSnapShot.SetValue(Grid.ColumnProperty, 1);
        //     //         BorderSnapShot.SetValue(Grid.ColumnSpanProperty, 1);
        //     //         BorderTrainingPlot.SetValue(Grid.ColumnProperty, 1);
        //     //         BorderTrainingPlot.SetValue(Grid.ColumnSpanProperty, 1);

        //     //         gridMain.RowDefinitions[0].Height = new GridLength(1.0, GridUnitType.Auto);
        //     //         gridMain.RowDefinitions[1].Height = new GridLength(20.0, GridUnitType.Pixel);
        //     //         gridMain.RowDefinitions[2].Height = new GridLength(1.0, GridUnitType.Star);

        //     //         BorderSnapShot.MaxHeight = 0.0;
        //     //         BorderSnapShot.UpdateLayout();

        //     //         Binding binding = new Binding
        //     //         {
        //     //             Path = new PropertyPath("ActualHeight"),
        //     //             Source = BorderWeightsMinMax
        //     //         };
        //     //         BorderSnapShot.SetBinding(Border.MaxHeightProperty, binding);

        //     //         BorderTrainingPlot.MaxHeight = 0.0;
        //     //         BorderTrainingPlot.UpdateLayout();

        //     //         binding = new Binding
        //     //         {
        //     //             Path = new PropertyPath("ActualHeight"),
        //     //             Source = BorderWeightsMinMax
        //     //         };
        //     //         BorderTrainingPlot.SetBinding(Border.MaxHeightProperty, binding);
        //     //         Zoomout = false;
        //          }

        //          args.Handled = true;
        //     }
        // }
}
