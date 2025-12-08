using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using Convnet.PageViewModels;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;


namespace Convnet.Dialogs
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
    public partial class TestParameters : Window
    {
        public DNNTrainingRate? Rate { get; set; }
        public DNNModel? Model { get; set; }
        public string? Path { get; set; }

        public TestPageViewModel? tpvm;
        public bool DialogResult { get; set; }

        public TestParameters()
        {
            DialogResult = false;

            InitializeComponent();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private void OnOpened(object? sender, System.EventArgs e)
        {
            if (Rate != null)
            {
                bool color = true;

                switch (Model?.Dataset)
                {
                    case DNNDatasets.cifar10:
                    case DNNDatasets.cifar100:
                    case DNNDatasets.tinyimagenet:
                        break;

                    case DNNDatasets.fashionmnist:
                    case DNNDatasets.mnist:
                        {
                            color = false;

                            var tbaa = this.FindControl<TextBox>("textBoxAutoAugment");
                            if (tbaa != null)
                                tbaa.IsEnabled = false;

                            var tbcc = this.FindControl<TextBox>("textBoxColorCast");
                            if (tbcc != null)
                                tbcc.IsEnabled = false;
                        }
                        break;
                }

                Rate.D = 1;
                Rate.PadD = 0;
                Rate.Cycles = 1;
                DataContext = Rate;

                var tbca = this.FindControl<TextBox>("textBoxColorAngle");
                if (tbca != null)
                    tbca.IsEnabled = Rate.ColorCast > 0 && color;

                var strategy = this.FindControl<CheckBox>("CheckBoxStrategy");
                if (strategy != null)
                    strategy.IsChecked = Settings.Default.UseTrainingStrategy;

                var tbc = this.FindControl<TextBox>("textBoxCycles");
                if (tbc != null)
                { 
                    tbc.Focus();
                    tbc.SelectAll();
                }
            }
        }


        private void ButtonTest_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            //if (IsValid(this))
            if (Model != null && Rate != null)
            {
                if (Model.BatchNormUsed() && Rate.N == 1)
                {
                    Dispatcher.UIThread.Post(() => MessageBox.Show("Your model uses batch normalization.\r\nThe batch size cannot be equal to 1 in this case.", "Warning", MessageBoxButtons.OK));
                    return;
                }
               
               
                DialogResult = true;
                Close();
            }
        }

        private void ButtonCancel_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            var bc = this.FindControl<Button>("buttonCancel");
            if (bc != null)
                bc.Focus();
        }

        private void TextBoxDistortions_TextChanged(object? sender, TextChangedEventArgs e)
        {
            var tbd = this.FindControl<TextBox>("textBoxDistortions");
            if (tbd != null)
            {
                var enabled = (float.TryParse(tbd.Text, out float result) && result > 0.0f);

                var cbi = this.FindControl<ComboBox>("comboBoInterpolation");
                var tbr = this.FindControl<TextBox>("textBoxRotation");
                var tbs = this.FindControl<TextBox>("textBoxScaling");

                if (cbi != null && tbr != null && tbs != null)
                {
                    cbi.IsEnabled = enabled;
                    tbr.IsEnabled = enabled;
                    tbs.IsEnabled = enabled;

                    e.Handled = true;
                }
            }
        }

        private void TextBoxColorCast_TextChanged(object? sender, TextChangedEventArgs e)
        {
            var tbca = this.FindControl<TextBox>("textBoxColorAngle");
            var tbcc = this.FindControl<TextBox>("textBoxColorCast");
            if (tbca != null && tbcc != null)
            {
                tbca.IsEnabled = (float.TryParse(tbcc.Text, out float result) && result > 0.0f);
                e.Handled = true;
            }
        }
    }
}