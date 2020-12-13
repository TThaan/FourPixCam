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
        internal static float Cost(Matrix a, Matrix t)
        {
            float result = 0;

            for (int j = 0; j < a.m; j++)
            {
                result += (t[j] - a[j]) * (t[j] - a[j]);//0.5f * 
            }

            return result;
            // Subtract(a, t, result);
            //result *= result;
            // throw new Exception();
            //SetHadamardProduct((t - a), (t - a), result); //
        }
        /// <summary>
        /// result = DCDA
        /// </summary>
        internal static void DerivationOfCostFunction(Matrix a, Matrix t, Matrix result)
        {
            Subtract(a, t, result);
            //result.ForEach(x => 2 * x);

            // Multiplicate(result, 2, result);

            // return (a - t);//
        }
    }
}
