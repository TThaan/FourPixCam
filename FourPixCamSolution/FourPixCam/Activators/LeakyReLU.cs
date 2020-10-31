using MatrixHelper;
using System.Linq;

namespace _4PixCam.Activators
{
    public class LeakyReLU : Activator
    {
        #region methods

        public override float GetValue(float z)
        {
            return z > 0
                ? z
                : z / 100;
        }
        public override Matrix GetValue(Matrix z)
        {
            return new Matrix(
                z.Select(x => x > 0f ? x : x/100)
                .ToArray());
        }
        public override float GetDerivativeWithRespectTo(float z)
        {
            return z > 0
                ? 1f
                : 1f / 100;
        }
        public override Matrix GetDerivativeWithRespectTo(Matrix z)
        {
            return new Matrix(
                z.Select(x => x > 0f ? 1f : 1f / 100)
                .ToArray()); 
        }

        #endregion
    }
}
