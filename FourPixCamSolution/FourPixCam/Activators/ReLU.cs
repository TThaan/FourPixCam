using MatrixHelper;

namespace FourPixCam.Activators
{
    public class ReLU
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input z.
        /// </summary>
        public static float Activation(float z)
        {
            return z >= 0
                ? z
                : 0;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        public static Matrix Activation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => x >= 0f ? x : 0);
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        public static float Derivation(float z)
        {
            return z >= 0
                ? 1
                : 0;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix Derivation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => x >= 0f ? 1f : 0);
        }
    }
}
