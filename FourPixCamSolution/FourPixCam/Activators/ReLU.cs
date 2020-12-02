using MatrixHelper;
using System.Linq;

namespace FourPixCam.Activators
{
    internal class ReLU
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input z.
        /// </summary>
        internal static float a(float z)
        {
            return z >= 0
                ? z
                : 0;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        internal static Matrix a(Matrix z)
        {
            return new Matrix(
                z.Select(x => x >= 0f ? x : 0)
                .ToArray());
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        internal static float dadz(float z)
        {
            return z >= 0
                ? 1
                : 0;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix dadz(Matrix z)
        {
            return new Matrix(z.Select(x => x >= 0f ? 1f : 0).ToArray())
                .Transpose; ;
        }
    }
}
