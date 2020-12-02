using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    internal class NullActivator// : Activation
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static float a(float z)
        {
            return z;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static Matrix a(Matrix z)
        {
            return z;
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static float dadz(float z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix dadz(Matrix z)
        {
            throw new NotImplementedException();
        }
    }
}
