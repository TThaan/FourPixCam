using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    internal class Sigmoid
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input neuron z.
        /// </summary>
        internal static float Activation(float z)
        {
            var tmp = 1 / (1 + (float)Math.Exp(-z));
            return tmp;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        internal static Matrix Activation(Matrix z)
        {
            return z.ForEach(x => 1 / (1 + (float)Math.Exp(-x)));
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        internal static float Derivation(float z)
        {
            var check = Activation(z) * (1 - Activation(z));
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
        internal static Matrix Derivation(Matrix z)
        {
            return z.ForEach(x => Activation(x) * (1 - Activation(x)));
        }
    }
}
