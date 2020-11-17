using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam.Activators
{
    public class Tanh
    {/// <summary>
     /// Activation ('squashing') function of any weighted input neuron z.
     /// </summary>
        public static float a(float z)
        {
            var tmp = (float)((Math.Exp(z)-Math.Exp(-z)) / (Math.Exp(z) + Math.Exp(-z)));
            return tmp;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        public static Matrix a(Matrix z)
        {
            return new Matrix(
                z.Select(z_j => (float)((Math.Exp(z_j) - Math.Exp(-z_j)) / (Math.Exp(z_j) + Math.Exp(-z_j))))
                .ToArray());
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        public static float dadz(float z)
        {
            var check = 1 - a(z) * a(z);
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
        public static Matrix dadz(Matrix z)
        {
            return new Matrix(z.Select(z_j => a(z_j) * (1 - a(z_j))).ToArray())
                .Transpose;
        }
    }
}
