using MatrixHelper;

namespace FourPixCam.Activators
{
    public class NullActivator// : Activation
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static float Activation(float z)
        {
            return z;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        public static Matrix Activation(Matrix result, Matrix z)
        {
            return result = result.ForEach(z, x => x);
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static float Derivation(float z)
        {
            // Check:
            return 1;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix Derivation(Matrix result, Matrix z)
        {
            // Check:
            return result.ForEach(x => 1);
        }
    }
}
