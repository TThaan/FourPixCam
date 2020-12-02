using MatrixHelper;

namespace FourPixCam.Activators
{
    internal class ReLU
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input z.
        /// </summary>
        internal static float Activation(float z)
        {
            return z >= 0
                ? z
                : 0;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        internal static Matrix Activation(Matrix z)
        {
            return z.ForEach(x => x >= 0f ? x : 0);
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        internal static float Derivation(float z)
        {
            return z >= 0
                ? 1
                : 0;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix Derivation(Matrix z)
        {
            return z.ForEach(x => x >= 0f ? 1f : 0);
        }
    }
}
