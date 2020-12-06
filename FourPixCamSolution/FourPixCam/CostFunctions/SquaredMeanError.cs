using MatrixHelper;

namespace FourPixCam.CostFunctions
{
    internal class SquaredMeanError
    {
        /// <summary>
        /// Cost/Error/Loss function of a single output neuron a.
        /// </summary>
        internal static float CostFunction(float a, float t)
        {
            return (t-a) * (t - a); //0.5f * 
        }
        /// <summary>
        /// Partial derivative of the cost with respect to a single output neuron a.
        /// </summary>
        internal static float DerivationOfCostFunction(float a, float t)
        {
            return (a - t);//2*
        }
        internal static Matrix CostFunction(Matrix a, Matrix t)
        {
            return t.Subtract(a) * t.Subtract(a);
            // return (t - a) * (t - a); //0.5f * 
        }
        internal static Matrix DerivationOfCostFunction(Matrix a, Matrix t)
        {
            return a.Subtract(t);
            // return (a - t);//2*
        }
    }
}
