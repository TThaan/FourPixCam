using MatrixHelper;

namespace FourPixCam.Activators
{
    /// <summary>
    /// Use this only on the output layer!
    /// </summary>
    internal class SoftMaxWithCrossEntropyLoss : SoftMax
    {
        new internal static Matrix Derivation(Matrix z)
        {
            // Check:
            return z.ForEach(x => 1);
        }
    }
}
