﻿namespace FourPixCam.CostFunctions
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
    }
}
