using Interop;
using System.Collections.ObjectModel;


namespace Convnet.Properties
{
    public sealed partial class Settings : global::System.Configuration.ApplicationSettingsBase
    {
        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingResult>? SelectedItems
        {
            get
            {
                return ((ObservableCollection<DNNTrainingResult>?)this[nameof(SelectedItems)]);
            }
            set
            {
                this[nameof(SelectedItems)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingResult>? TrainingLog
        {
            get
            {
                return ((ObservableCollection<DNNTrainingResult>?)this[nameof(TrainingLog)]);
            }
            set
            {
                this[nameof(TrainingLog)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingStrategy>? TrainingStrategies
        {
            get
            {
                return ((ObservableCollection<DNNTrainingStrategy>?)this[nameof(TrainingStrategies)]);
            }
            set
            {
                this[nameof(TrainingStrategies)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingRate>? TrainingRates
        {
            get
            {
                return ((ObservableCollection<DNNTrainingRate>?)this[nameof(TrainingRates)]);
            }
            set
            {
                this[nameof(TrainingRates)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public DNNTrainingRate? TraininingRate
        {
            get
            {
                return ((DNNTrainingRate?)this[nameof(TraininingRate)]);
            }
            set
            {
                this[nameof(TraininingRate)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public DNNTrainingRate? TestRate
        {
            get
            {
                return ((DNNTrainingRate?)this[nameof(TestRate)]);
            }
            set
            {
                this[nameof(TestRate)] = value;
            }
        }
    }
}
