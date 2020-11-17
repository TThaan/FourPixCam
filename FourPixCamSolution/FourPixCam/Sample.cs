using MatrixHelper;
using System;

namespace FourPixCam
{
    public class Sample
    {
        public Label Label { get; set; }
        public Matrix RawInput;
        public Matrix Input;
        public Matrix ExpectedOutput;
        // public Matrix ActualOutput { get; set; }

        // Returns true if each output equals or is close to each expectedOutput.
        public Func<Matrix, float, bool> IsOutputCorrect => IsOutputApproximatelyCorrect;
            // outputMatrix => outputMatrix == ExpectedOutput;

        #region helpers

        bool IsOutputApproximatelyCorrect(Matrix output, float tolerance)
        {
            if (output.m == ExpectedOutput.m && output.n == 1 && ExpectedOutput.n == 1)
            {
                for (int j = 0; j < output.m; j++)
                {
                    var a_j = output[j];
                    var t_j = ExpectedOutput[j];
                    var x = Math.Abs(t_j - a_j);
                    if (x > tolerance)
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
