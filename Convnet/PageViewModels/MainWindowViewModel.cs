using ReactiveUI;


namespace ConvnetAvalonia.PageViewModels
{
    public class MainWindowViewModel : ReactiveObject
    {
        public PageViewModel? PageVM;

        //public ReactiveCommand<Unit, Unit> CutCommand { get; }

        //private bool canCut = true;

        //public bool CanCut
        //{
        //    get => ApplicationCommands.Cut.CanExecute(null, null);
        //    set => this.RaiseAndSetIfChanged(ref canCut, value);
        //}

        public MainWindowViewModel()
        {
            //CutCommand = ReactiveCommand.Create(Cut, this.WhenAnyValue(x => x.CanCut));
        }


        //public void Cut()
        //{
        //    //var elem = TopLevel.GetTopLevel(this).GetFocusedElement();
        //    ApplicationCommands.Cut.Execute(null, null);

        //    //if (PageVM != null && PageVM.Pages != null)
        //    //{
        //    //    //var epvm = MainView.PageViews.Items[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;

        //    //    var epvm = PageVM.Pages[(int)PageViewModels.ViewModels.Edit] as EditPageViewModel;
        //    //    if (epvm != null)
        //    //    {
        //    //        var topLevel = TopLevel.GetTopLevel(this);
        //    //        if (FocusManager != null)
        //    //        {
        //    //            var elem = FocusManager.GetFocusedElement();

        //    //        }
        //    //    }
        //    //}
        //}
    }
}
