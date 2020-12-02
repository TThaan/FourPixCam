using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam.Activators
{
    internal class Sigmoid
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input neuron z.
        /// </summary>
        internal static float a(float z)
        {
            var tmp = 1 / (1 + (float)Math.Exp(-z));
            return tmp;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        internal static Matrix a(Matrix z)
        {
            return new Matrix(
                z.Select(x => 1 / (1 + (float)Math.Exp(-x)))
                .ToArray());
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        internal static float dadz(float z)
        {
            var check = a(z) * (1 - a(z));
            if (float.IsInfinity(check))
            {

            }
            if (float.IsNaN(check))
            {

            }
            return check;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix dadz(Matrix z)
        {
            return new Matrix(z.Select(z_j => a(z_j) * (1 - a(z_j))).ToArray())
                .Transpose;
        }
    }
}
