using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using Convnet.Common;
using Convnet.PageViewModels;
using Convnet.Properties;
using CustomMessageBox.Avalonia;
using Interop;
using System.Collections.ObjectModel;


namespace Convnet.Dialogs
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "<Pending>")]
    public partial class TrainParameters : Window
    {
        public DNNTrainingRate Rate { get; set; }
        public DNNModel Model { get; set; }
        public string Path { get; set; }

        public TrainPageViewModel tpvm;
        public bool DialogResult { get; set; }

        public TrainParameters()
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

                switch (Model.Dataset)
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

                var tbge = this.FindControl<TextBox>("textBoxGotoEpoch");
                if (tbge != null)
                {
                    tbge.Text = tpvm.GotoEpoch.ToString(); ;
                }

                var sgdr = this.FindControl<CheckBox>("CheckBoxSGDR");
                if (sgdr != null)
                {
                    sgdr.IsChecked = tpvm.SGDR;
                    ChangeSGDR();
                }

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


        //void Window_Loaded(object sender, EventArgs e)
        //{
        //    switch (Model.Dataset)
        //    {
        //        case DNNDatasets.cifar10:
        //        case DNNDatasets.cifar100:
        //        case DNNDatasets.tinyimagenet:
        //            break;

        //        case DNNDatasets.fashionmnist:
        //        case DNNDatasets.mnist:
        //            Rate.AutoAugment = 0.0f;
        //            Rate.ColorCast = 0;
        //            Rate.ColorAngle = 0;
        //            textBoxAutoAugment.IsEnabled = false;
        //            textBoxColorCast.IsEnabled = false;
        //            textBoxColorAngle.IsEnabled = false;
        //            break;
        //    }

        //    Rate.D = 1;
        //    Rate.PadD = 0;
        //    Rate.Cycles = 1;
        //    DataContext = Rate;

        //    textBoxGotoEpoch.Text = tpvm.GotoEpoch.ToString();
        //    CheckBoxSGDR.IsChecked = tpvm.SGDR;
        //    ChangeSGDR();

        //    textBoxColorAngle.IsEnabled = Rate.ColorCast > 0;

        //    textBoxCycles.Focus();
        //    //textBoxCycles.Select(0, textBoxCycles.GetLineLength(0));

        //    CheckBoxStrategy.IsChecked = Settings.Default.UseTrainingStrategy;
        //}

        //private bool IsValid(DependencyObject node)
        //{
        //    // Check if dependency object was passed
        //    if (node != null)
        //    {
        //        // Check if dependency object is valid.
        //        // NOTE: Validation.GetHasError works for controls that have validation rules attached 
        //        bool isValid = !Validation.GetHasError(node);
        //        if (!isValid)
        //        {
        //            // If the dependency object is invalid, and it can receive the focus,
        //            // set the focus
        //            if (node is IInputElement) Keyboard.Focus((IInputElement)node);
        //            return false;
        //        }
        //    }

        //    // If this dependency object is valid, check all child dependency objects
        //    foreach (object subnode in LogicalTreeHelper.GetChildren(node))
        //    {
        //        if (subnode is DependencyObject)
        //        {
        //            // If a child dependency object is invalid, return false immediately,
        //            // otherwise keep checking
        //            if (IsValid((DependencyObject)subnode) == false) return false;
        //        }
        //    }

        //    // All dependency objects are valid
        //    return true;
        //}

        private void ButtonTrain_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            //if (IsValid(this))
            {
                if (Model.BatchNormUsed() && Rate.N == 1)
                {
                    Dispatcher.UIThread.Post(() => MessageBox.Show("Your model uses batch normalization.\r\nThe batch size cannot be equal to 1 in this case.", "Warning", MessageBoxButtons.OK));
                    return;
                }
                var tb = this.FindControl<TextBox>("textBoxGotoEpoch");
                if (tb != null)
                {
                    uint.TryParse(tb.Text, out uint gotoEpoch);
                    if ((gotoEpoch > (tpvm.SGDR ? Rate.Epochs * Rate.Cycles * Rate.EpochMultiplier : Rate.Epochs)) || (gotoEpoch < 1))
                    {
                        Dispatcher.UIThread.Post(() => MessageBox.Show("Goto epoch is to large", "Warning", MessageBoxButtons.OK));
                        return;
                    }
                    tpvm.GotoEpoch = gotoEpoch;
                    tpvm.GotoCycle = 1;
                }

                Settings.Default.TraininingRate = Rate;
                Settings.Default.Optimizer = (int)Rate.Optimizer;
                Settings.Default.Save();

                Model.ClearTrainingStrategies();
                foreach (DNNTrainingStrategy strategy in Settings.Default.TrainingStrategies)
                    Model.AddTrainingStrategy(strategy);
                Model.TrainingStrategies = Settings.Default.TrainingStrategies;

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

        private void CheckBoxSGDR_Checked(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            ChangeSGDR();
        }

        private void CheckBoxStrategy_Checked(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            var cbs = this.FindControl<CheckBox>("CheckBoxStrategy");
            if (cbs != null)
            {
                var useStrategy = cbs.IsChecked.HasValue && cbs.IsChecked.Value;
                tpvm.Model.SetUseTrainingStrategy(useStrategy);
                Settings.Default.UseTrainingStrategy = useStrategy;
                Settings.Default.Save();

                e.Handled = true;
            }
        }

        private void CheckBoxStrategy_Unchecked(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            var cbs = this.FindControl<CheckBox>("CheckBoxStrategy");
            if (cbs != null)
            {
                var useStrategy = cbs.IsChecked.HasValue && cbs.IsChecked.Value;
                tpvm.Model.SetUseTrainingStrategy(useStrategy);
                Settings.Default.UseTrainingStrategy = useStrategy;
                Settings.Default.Save();

                e.Handled = true;
            }
        }

        private void ChangeSGDR()
        {
            var sgdr = this.FindControl<CheckBox>("CheckBoxSGDR");
            if (sgdr != null)
                tpvm.SGDR = sgdr.IsChecked.HasValue ? sgdr.IsChecked.Value : false;

            var tbc = this.FindControl<TextBox>("textBoxCycles");
            if (tbc != null)
                tbc.IsEnabled = tpvm.SGDR;

            var tbem = this.FindControl<TextBox>("textBoxEpochMultiplier");
            if (tbem != null)
                tbem.IsEnabled = tpvm.SGDR;

            var tbdf = this.FindControl<TextBox>("textBoxDecayFactor");
            if (tbdf != null)
                tbdf.IsEnabled = !tpvm.SGDR;

            var tbae = this.FindControl<TextBox>("textBoxDecayAfterEpochs");
            if (tbae != null)
                tbae.IsEnabled = !tpvm.SGDR;
        }

        private void comboBoOptimizer_SelectionChanged(object? sender, SelectionChangedEventArgs e)
        {
            DNNOptimizers optimizer = (DNNOptimizers)comboBoOptimizer.SelectedIndex;

            switch (optimizer)
            {
                case DNNOptimizers.SGD:
                    {
                        textBlockMomentum.Opacity = 0.5;
                        textBoxMomentum.IsEnabled = false;
                        textBlockL2penalty.Opacity = 1;
                        textBoxL2penalty.IsEnabled = true;
                    }
                    break;

                case DNNOptimizers.AdaBelief:
                case DNNOptimizers.AdaBound:
                case DNNOptimizers.AdaDelta:
                case DNNOptimizers.Adam:  
                case DNNOptimizers.Adamax:
                case DNNOptimizers.AmsBound:
                case DNNOptimizers.RMSProp:
                    {
                        textBlockMomentum.Opacity = 1;
                        textBoxMomentum.IsEnabled = true;
                        textBlockL2penalty.Opacity = 0.5;
                        textBoxL2penalty.IsEnabled = false;
                    }
                    break;

                case DNNOptimizers.AdaGrad:
                    {
                        textBlockMomentum.Opacity = 0.5;
                        textBoxMomentum.IsEnabled = false;
                        textBlockL2penalty.Opacity = 0.5;
                        textBoxL2penalty.IsEnabled = false;
                    }
                    break;

                case DNNOptimizers.AdaBoundW:
                case DNNOptimizers.AdamW:
                case DNNOptimizers.AmsBoundW:
                case DNNOptimizers.NAG:
                case DNNOptimizers.SGDMomentum:
                case DNNOptimizers.SGDW:
                    {
                        textBlockMomentum.Opacity = 1;
                        textBoxMomentum.IsEnabled = true;
                        textBlockL2penalty.Opacity = 1;
                        textBoxL2penalty.IsEnabled = true;
                    }
                    break;
            }

            switch (optimizer)
            {
                case DNNOptimizers.AdaDelta:
                case DNNOptimizers.AdaGrad:
                case DNNOptimizers.NAG:
                case DNNOptimizers.SGD:
                case DNNOptimizers.SGDMomentum:
                case DNNOptimizers.SGDW:
                    {
                        textBlockBeta2.Opacity = 0.5;
                        textBoxBeta2.IsEnabled = false;
                    }
                    break;

                case DNNOptimizers.AdaBelief:
                case DNNOptimizers.AdaBound:
                case DNNOptimizers.AdaBoundW:
                case DNNOptimizers.Adam:
                case DNNOptimizers.AdamW:
                case DNNOptimizers.Adamax:
                case DNNOptimizers.AmsBound:
                case DNNOptimizers.AmsBoundW:
                case DNNOptimizers.RMSProp:
                    {
                        textBlockBeta2.Opacity = 1;
                        textBoxBeta2.IsEnabled = true;
                    }
                    break;
            }

            switch (optimizer)
            {
                case DNNOptimizers.AdaBelief:
                case DNNOptimizers.AdaDelta:
                case DNNOptimizers.AdaGrad:
                case DNNOptimizers.Adam:
                case DNNOptimizers.AdamW:
                case DNNOptimizers.Adamax:
                case DNNOptimizers.RMSProp:
                case DNNOptimizers.NAG:
                case DNNOptimizers.SGD:
                case DNNOptimizers.SGDMomentum:
                case DNNOptimizers.SGDW:
                    {
                        textBlockFinalLR.Opacity = 0.5;
                        textBoxFinalRate.IsEnabled = false;
                        textBlockGamma.Opacity = 0.5;
                        textBoxGamma.IsEnabled = false;
                    }
                    break;

                case DNNOptimizers.AdaBound:
                case DNNOptimizers.AdaBoundW:
                case DNNOptimizers.AmsBound:
                case DNNOptimizers.AmsBoundW:
                    {
                        textBlockFinalLR.Opacity = 1;
                        textBoxFinalRate.IsEnabled = true;
                        textBlockGamma.Opacity = 1;
                        textBoxGamma.IsEnabled = true;
                    }
                    break;
            }
        }

        private void ButtonSGDRHelp_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            ApplicationHelper.OpenBrowser("https://arxiv.org/abs/1608.03983");
        }

        private void ButtonStrategies_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        {
            if (Settings.Default.TrainingStrategies == null)
            {
                var rate = Rate;
                var strategy = new DNNTrainingStrategy(1.0f, rate.N, rate.D, rate.H, rate.W, rate.PadD, rate.PadH, rate.PadW, rate.Momentum, rate.Beta2, rate.Gamma, rate.L2Penalty, rate.Dropout, rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation);
                Settings.Default.TrainingStrategies = new ObservableCollection<DNNTrainingStrategy> { strategy };
                Settings.Default.Save();
            }

            tpvm.TrainingStrategies = Settings.Default.TrainingStrategies;

            //TrainingStrategiesEditor dialog = new TrainingStrategiesEditor { Path = TrainPageViewModel.StorageDirectory };
            //dialog.tpvm = tpvm;
            //dialog.DataContext = tpvm;
            //dialog.buttonOk.IsEnabled = true;
            ////dialog.Owner = Application.Current.MainWindow;
            //dialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;

            //if (dialog.ShowDialog() ?? false)
            //{
            //    Settings.Default.TrainingStrategies = tpvm.TrainingStrategies;
            //    Settings.Default.Save();

            //    Model.ClearTrainingStrategies();
            //    foreach (DNNTrainingStrategy strategy in tpvm.TrainingStrategies)
            //        Model.AddTrainingStrategy(strategy);
            //}
        }
    }
}