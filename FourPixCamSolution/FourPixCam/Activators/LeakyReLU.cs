using MatrixHelper;

namespace FourPixCam.Activators
{
    internal class LeakyReLU
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static float Activation(float z)
        {
            return z >= 0
                ? z
                : z / 100;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static Matrix Activation(Matrix z)
        {
            return z.ForEach(x => x >= 0f ? x : x/100);
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static float Derivation(float z)
        {
            return z >= 0
                ? 1f
                : 1f / 100;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix Derivation(Matrix z)
        {
            return z.ForEach(x => x >= 0f ? 1f : 1f / 100);
        }
    }
}
