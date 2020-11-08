using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam.CostFunctions
{
    class SquaredMeanError
    {
        /// <summary>
        /// Error/Cost/Loss Matrix
        /// </summary>
        public static Matrix E(Matrix a, Matrix t)
        {
            if (a.n != 1 || t.n != 1)
            {
                throw new ArgumentException("The output matrix a as well as the expected-output matrix can only have one column.");
            }
            Matrix gap = t - a;
            var tmp = .5f * Operations.HadamardProduct(gap, gap);   // Overloaded method in MatrixHelper for hadamard-square?
            return tmp;
        }
        /// <summary>
        /// Half the squared difference between expected and actual output.
        /// </summary>
        public static float E(float a, float t)
        {
            return 0.5f * (t-a) * (t - a);
        }
        /// <summary>
        /// Half the squared difference between expected and actual output.
        /// </summary>
        public static float ETotal(Matrix a, Matrix t)
        {
            // ETotal = Sum or Sum divided by a.m?
            return E(a, t).Sum()/a.m;
        }
        /// <summary>
        /// Partial derivative of the cost
        /// with regards to the output, i.e. a^L.
        /// </summary>
        public static float dEda(float a, float t)
        {
            return a - t;
        }
        /// <summary>
        /// Partial derivative of the cost
        /// with regards to the output, i.e. a^L.
        /// </summary>
        public static Matrix dEda(Matrix a, Matrix t)
        {
            if (a.n != 1 || t.n != 1)
            {
                throw new ArgumentException("The output matrix a as well as the expected-output matrix can only have one column.");
            }
            return a - t;
        }
    }
}
