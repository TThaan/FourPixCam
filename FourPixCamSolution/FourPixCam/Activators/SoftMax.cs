using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam.Activators
{
    public class SoftMax// : Activation
    {
        #region methods

        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static float Activation(float z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static Matrix Activation(Matrix result, Matrix z)
        {
            result = z.ForEach(x => (float)Math.Exp(x));
            float sum = result.Sum();
            return sum == 0 ? result : result / sum;
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static float Derivation(float z)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix Derivation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => x * (1 - x));
        }

        #endregion
    }
}
