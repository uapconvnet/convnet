using Interop;
using System.Collections.ObjectModel;

namespace Convnet.Properties
{
    public sealed partial class Settings : global::System.Configuration.ApplicationSettingsBase
    {
        [global::System.Configuration.UserScopedSetting()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingResult>? TrainingLog
        {
            get
            {
                return this[nameof(TrainingLog)] as ObservableCollection<DNNTrainingResult>;
            }
            set
            {
                this[nameof(TrainingLog)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingStrategy>? TrainingStrategies
        {
            get
            {
                return this[nameof(TrainingStrategies)] as ObservableCollection<DNNTrainingStrategy>;
            }
            set
            {
                this[nameof(TrainingStrategies)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public ObservableCollection<DNNTrainingRate>? TrainingRates
        {
            get
            {
                return this[nameof(TrainingRates)] as ObservableCollection<DNNTrainingRate>;
            }
            set
            {
                this[nameof(TrainingRates)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public DNNTrainingRate? TraininingRate
        {
            get
            {
                return this[nameof(TraininingRate)] as DNNTrainingRate;
            }
            set
            {
                this[nameof(TraininingRate)] = value;
            }
        }

        [global::System.Configuration.UserScopedSetting()]
        [global::System.Configuration.SettingsSerializeAs(global::System.Configuration.SettingsSerializeAs.Xml)]
        public DNNTrainingRate? TestRate
        {
            get
            {
                return this[nameof(TestRate)] as DNNTrainingRate;
            }
            set
            {
                this[nameof(TestRate)] = value;
            }
        }
    }
}
