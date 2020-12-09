using MatrixHelper;
using System;
using static MatrixHelper.Operations2;

namespace FourPixCam.CostFunctions
{
    internal class SquaredMeanError
    {
        /// <summary>
        /// Cost/Error/Loss function of a single output neuron a.
        /// </summary>
        internal static float CostFunction(float a, float t)
        {
            return (t - a) * (t - a); //0.5f * 
        }
        /// <summary>
        /// Partial derivative of the cost with respect to a single output neuron a.
        /// </summary>
        internal static float DerivationOfCostFunction(float a, float t)
        {
            return (a - t);//2*
        }
        /// <summary>
        /// result = DCDA
        /// </summary>
        internal static void CostFunction(Matrix a, Matrix t, Matrix result)
        {
            for (int j = 0; j < result.m; j++)
            {
                result[j] = (t[j] - a[j]) * (t[j] - a[j]);
            }
            // Subtract(a, t, result);
            //result *= result;
            // throw new Exception();
            //SetHadamardProduct((t - a), (t - a), result); //0.5f * 
        }
        /// <summary>
        /// result = DCDA
        /// </summary>
        internal static void DerivationOfCostFunction(Matrix a, Matrix t, Matrix result)
        {
            Subtract(a, t, result);
            // return (a - t);//2*
        }
    }
}
