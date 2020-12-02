using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    internal class Tanh
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input neuron z.
        /// </summary>
        internal static float Activation(float z)
        {
            var tmp = (float)((Math.Exp(z)-Math.Exp(-z)) / (Math.Exp(z) + Math.Exp(-z)));
            var test = 1 - tmp * tmp;
            return tmp;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        internal static Matrix Activation(Matrix z)
        {
            return z.ForEach(x => (float)((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))));
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        internal static float Derivation(float z)
        {
            var result = 1 - Activation(z) * Activation(z);
            //if (float.IsInfinity(result))
            //{

            //}
            //if (float.IsNaN(result))
            //{

            //}
            return result;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix Derivation(Matrix z)
        {
            return z.ForEach(x => 1 - Activation(x) * Activation(x));
        }
    }
}
