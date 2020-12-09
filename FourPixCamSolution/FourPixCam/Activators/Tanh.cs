using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    public class Tanh
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input neuron z.
        /// </summary>
        public static float Activation(float z)
        {
            var tmp = (float)((Math.Exp(z)-Math.Exp(-z)) / (Math.Exp(z) + Math.Exp(-z)));
            // var test = 1 - tmp * tmp;
            return tmp;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        public static Matrix Activation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => (float)((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))));
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        public static float Derivation(float z)
        {
            var result = 1 - (Activation(z) * Activation(z));
            return result;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix Derivation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => Derivation(x));
        }
    }
}
