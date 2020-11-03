using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    public class Sigmoid// : Activation
    {
        #region methods

        public static float f(float z)
        {
            return 1 / (1 + (float)Math.Exp(-z));
        }
        public static Matrix f(Matrix z)
        {
            throw new NotImplementedException();
        }
        public static float df(float z)
        {
            return z * (1 - z);
        }
        public static Matrix df(Matrix z)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
