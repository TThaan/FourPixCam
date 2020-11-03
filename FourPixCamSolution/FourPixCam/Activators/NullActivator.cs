using MatrixHelper;
using System;

namespace FourPixCam.Activators
{
    class NullActivator// : Activation
    {
        public static float f(float z)
        {
            return z;
        }
        public static Matrix f(Matrix z)
        {
            return z;
        }

        public static float df(float z)
        {
            throw new NotImplementedException();
        }
        public static Matrix df(Matrix z)
        {
            throw new NotImplementedException();
        }
    }
}
