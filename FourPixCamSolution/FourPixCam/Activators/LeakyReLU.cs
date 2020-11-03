using MatrixHelper;
using System.Linq;

namespace FourPixCam.Activators
{
    public class LeakyReLU// : Activation
    {
        #region methods

        public static float f(float z)
        {
            return z > 0
                ? z
                : z / 100;
        }
        public static Matrix f(Matrix z)
        {
            return new Matrix(
                z.Select(x => x > 0f ? x : x/100)
                .ToArray());
        }
        public static float df(float z)
        {
            return z > 0
                ? 1f
                : 1f / 100;
        }
        public static Matrix df(Matrix z)
        {
            return new Matrix(
                z.Select(x => x > 0f ? 1f : 1f / 100)
                .ToArray()); 
        }

        #endregion
    }
}
