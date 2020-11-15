using MatrixHelper;
using System.Linq;

namespace FourPixCam.Activators
{
    public class LeakyReLU
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static float a(float z)
        {
            return z >= 0
                ? z
                : z / 100;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static Matrix a(Matrix z)
        {
            return new Matrix(
                z.Select(x => x >= 0f ? x : x/100)
                .ToArray());
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static float dadz(float z)
        {
            return z >= 0
                ? 1f
                : 1f / 100;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix dadz(Matrix z)
        {
            return new Matrix(
                z.Select(x => x >= 0f ? 1f : 1f / 100).ToArray())
                .Transpose; ; 
        }
    }
}
