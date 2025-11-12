using Interop;
using System;
using System.Globalization;
using Avalonia.Controls;

namespace Convnet.Common
{
    public class UIntRangeRule : ValidationRule
    {
        public uint Min { get; set; }
        public uint Max { get; set; }

        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            uint intValue = 0;

            try
            {
                string str = value as string;
                if (str.Length > 0)
                    intValue = uint.Parse(str, cultureInfo);
            }
            catch (Exception e)
            {
                return new ValidationResult(false, e.Message);
            }

            if ((intValue < Min) || (intValue > Max))
                return new ValidationResult(false, "Please enter a value in the range: " + Min + " - " + Max + ".");

            return new ValidationResult(true, null);
        }
    }

    public class UIntBatchSizeRangeRule : ValidationRule
    {
        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            uint intValue = 0;

            try
            {
                string str = value as string;
                if (str.Length > 0)
                    intValue = uint.Parse(str, cultureInfo);
            }
            catch (Exception e)
            {
                return new ValidationResult(false, e.Message);
            }

            if (DNNModel.StochasticEnabled())
            {
                if (intValue < 1u)
                    return new ValidationResult(false, "Please enter one or a multiple of eight.");

                if (intValue > 1u && intValue % 8u != 0)
                    return new ValidationResult(false, "Please enter one or a multiple of eight.");
            }
            else
            {
                if (intValue < 8u)
                    return new ValidationResult(false, "Please enter a multiple of eight.");

                if (intValue % 8u != 0)
                    return new ValidationResult(false, "Please enter a multiple of eight.");
            }

            return new ValidationResult(true, null);
        }
    }

    public class FloatRangeRule : ValidationRule
    {
        public float Min { get; set; }
        public float Max { get; set; }

        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            float floatValue = 0;

            try
            {
                string str = value as string;
                if (str.Length > 0)
                    floatValue = float.Parse(str, cultureInfo);
            }
            catch (Exception e)
            {
                return new ValidationResult(false, e.Message);
            }

            if ((floatValue < Min) || (floatValue > Max))
                return new ValidationResult(false, "Please enter a value in the range: " + Min + " - " + Max + ".");
            else
                return new ValidationResult(true, null);
        }
    }
}
