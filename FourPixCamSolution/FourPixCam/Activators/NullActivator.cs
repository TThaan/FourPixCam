using MatrixHelper;
using System;

namespace _4PixCam.Activators
{
    class NullActivator : Activator
    {
        public override float GetValue(float z)
        {
            return z;
        }
        public override Matrix GetValue(Matrix z)
        {
            return z;
        }

        public override float GetDerivativeWithRespectTo(float z)
        {
            throw new NotImplementedException();
        }
        public override Matrix GetDerivativeWithRespectTo(Matrix z)
        {
            throw new NotImplementedException();
        }
    }
}
