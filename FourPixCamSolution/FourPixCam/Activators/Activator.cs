using MatrixHelper;

namespace _4PixCam.Activators
{
    public abstract class Activator
    {
        /// <summary>
        /// Function..
        /// </summary>
        /// <param name="z">totalInput</param>
        /// <returns></returns>
        public abstract float GetValue(float z);
        public abstract Matrix GetValue(Matrix z);
        /// <summary>
        /// Derivative of Activator.Function().
        /// </summary>
        public abstract float GetDerivativeWithRespectTo(float z);
        public abstract Matrix GetDerivativeWithRespectTo(Matrix z);
    }
}
