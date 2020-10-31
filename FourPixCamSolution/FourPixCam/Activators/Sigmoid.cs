using MatrixHelper;
using System;

namespace _4PixCam.Activators
{
    public class Sigmoid : Activator
    {
        #region methods

        public override float GetValue(float z)
        {
            return 1 / (1 + (float)Math.Exp(-z));
        }
        public override Matrix GetValue(Matrix z)
        {
            throw new NotImplementedException();
        }
        public override float GetDerivativeWithRespectTo(float z)
        {
            return z * (1 - z);
        }
        public override Matrix GetDerivativeWithRespectTo(Matrix z)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
