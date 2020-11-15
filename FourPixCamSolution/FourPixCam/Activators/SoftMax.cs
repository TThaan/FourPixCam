using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    public class SoftMax// : Activation
    {
        #region methods

        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static float a(float z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static Matrix a(Matrix z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static float dadz(float z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix dadz(Matrix z)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
