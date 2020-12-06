using MatrixHelper;

namespace FourPixCam.Activators
{
    /// <summary>
    /// Use this only on the output layer!
    /// </summary>
    public class SoftMaxWithCrossEntropyLoss : SoftMax
    {
        new public static Matrix Derivation(Matrix result, Matrix z)
        {
            // Check:
            return result.ForEach(z, x => 1);
        }
    }
}
