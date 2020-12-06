using MatrixHelper;

namespace FourPixCam.Activators
{
    public class LeakyReLU
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static float Activation(float z)
        {
            return z >= 0
                ? z
                : z / 100;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static Matrix Activation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => x >= 0f ? x : x/100);
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static float Derivation(float z)
        {
            return z >= 0
                ? 1f
                : 1f / 100;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix Derivation(Matrix result, Matrix z)
        {
            return result.ForEach(z, x => x >= 0f ? 1f : 1f / 100);
        }
    }
}
