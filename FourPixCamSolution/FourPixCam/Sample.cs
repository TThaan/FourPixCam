using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam
{
    public class Sample
    {
        public Matrix Input;
        public Matrix ExpectedOutput;

        // Returns true if each output equals each expectedOutput.
        public Func<Matrix, bool> IsOutputCorrect => 
            outputMatrix => outputMatrix.Select((singleOut, index) => singleOut == ExpectedOutput[index, 0])
            .All(x => x == true);
    }
}
