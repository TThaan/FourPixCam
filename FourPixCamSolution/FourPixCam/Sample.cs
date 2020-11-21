﻿using MatrixHelper;
using System;

namespace FourPixCam
{
    public class Sample
    {
        #region fields

        bool? isOutputCorrect;
        Matrix actualOutput;

        #endregion

        #region public

        public static float Tolerance { get; set; }

        public Label Label { get; set; }
        public Matrix RawInput { get; set; }
        public Matrix Input { get; set; }
        public Matrix ExpectedOutput { get; set; }

        public Matrix ActualOutput 
        {
            get => actualOutput;
            set
            {
                // Always when 'ActualOutput' is (re)set change 'isOutputCorrect' to null.
                isOutputCorrect = null;
                actualOutput = value;
            }
        }
        public bool? IsOutputCorrect
        {
            get
            {
                if (ActualOutput != null)
                {
                    if (isOutputCorrect != null)
                    {
                        return isOutputCorrect;
                    }
                    else
                    {
                        isOutputCorrect = IsOutputApproximatelyCorrect();
                        return isOutputCorrect;
                    }
                }
                else
                {
                    throw new ArgumentException("You cannot check the output when there is no actual output.");
                }
            }
        }

        #endregion

        #region helpers

        bool? IsOutputApproximatelyCorrect()
        {
            if (ActualOutput.m == ExpectedOutput.m && ActualOutput.n == 1 && ExpectedOutput.n == 1)
            {
                for (int j = 0; j < ActualOutput.m; j++)
                {
                    var a_j = ActualOutput[j];
                    var t_j = ExpectedOutput[j];
                    var x = Math.Abs(t_j - a_j);
                    if (x > Tolerance)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        #endregion
    }
}
