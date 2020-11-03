using MatrixHelper;
using System.Linq;

namespace FourPixCam.CostFunctions
{
    class SquaredMeanError
    {
        public static float C(float a, float t)
        {
            return 0.5f * (t-a) * (t - a);
        }
        public static Matrix C(Matrix a, Matrix t)
        {
            Matrix gap = t - a;
            Matrix gapSquared = Operations.HadamardProduct(gap, gap);   // Wwn overloaded method in MatrixHelper for hadamard-square?
            return gapSquared;// 0.5f * (t - a) * (t - a);
        }   
        public static float dC(float a, float t)
        {
            return a - t;
        }
        public static Matrix dC(Matrix a, Matrix t)
        {
            return a - t;
        }
    }
}
