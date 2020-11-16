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
        public Func<Matrix, bool> IsOutputCorrect => IsOutputApproximatelyCorrect;
            // outputMatrix => outputMatrix == ExpectedOutput;

        #region helpers

        bool IsOutputApproximatelyCorrect(Matrix output)
        {
            if (output.m == ExpectedOutput.m && output.n == 1 && ExpectedOutput.n == 1)
            {
                for (int j = 0; j < output.m; j++)
                {
                    if (ExpectedOutput[j] == 0)
                    {
                        var x = output[j];
                        return x < 0.1;
                    }
                    else if (ExpectedOutput[j] == 1)
                    {
                        var x = output[j];
                        return x >= 0.1;
                    }
                    else
                    {
                        throw new ArgumentException("Unexpected ExpectedOutput!");
                    }
                }
            }
            return false;
        }

        #endregion
    }
}
