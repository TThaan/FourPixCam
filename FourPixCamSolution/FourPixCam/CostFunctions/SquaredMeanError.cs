using MatrixHelper;
using System.Linq;

namespace FourPixCam.CostFunctions
{
    class SquaredMeanError
    {
        /// <summary>
        /// Half the squared difference between expected and actual output.
        /// </summary>
        public static float C(float a, float t)
        {
            return 0.5f * (t-a) * (t - a);
        }
        /// <summary>
        /// Half the squared difference between expected and actual output.
        /// </summary>
        public static float C(Matrix a, Matrix t)
        {
            Matrix gap = t - a;
            Matrix gapSquared = Operations.HadamardProduct(gap, gap);   // Overloaded method in MatrixHelper for hadamard-square?
            return gapSquared.Sum();// 0.5f * (t - a) * (t - a);
        }
        /// <summary>
        /// Partial derivative of the cost
        /// with regards to the output, i.e. a^L.
        /// </summary>
        public static float dCda(float a, float t)
        {
            return a - t;
        }
        /// <summary>
        /// Partial derivative of the cost
        /// with regards to the output, i.e. a^L.
        /// </summary>
        public static Matrix dCda(Matrix a, Matrix t)
        {
            return a - t;
        }
    }
}
