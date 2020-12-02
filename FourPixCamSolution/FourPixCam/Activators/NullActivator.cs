using MatrixHelper;

namespace FourPixCam.Activators
{
    internal class NullActivator// : Activation
    {
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static float Activation(float z)
        {
            return z;
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input z.
        /// </summary>
        internal static Matrix Activation(Matrix z)
        {
            return z;
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static float Derivation(float z)
        {
            // Check:
            return 1;
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        internal static Matrix Derivation(Matrix z)
        {
            // Check:
            return z.ForEach(x => 1);
        }
    }
}
