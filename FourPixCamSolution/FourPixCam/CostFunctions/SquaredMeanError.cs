namespace FourPixCam.CostFunctions
{
    /// <summary>
    /// Only (elements`) function & derivative should be in here,
    /// no matrix operations!?! Those belong to NeurNetMath.
    /// </summary>
    class SquaredMeanError
    {
        /// <summary>
        /// Cost/Error/Loss function of a single output neuron a.
        /// </summary>
        public static float CostFunction(float a, float t)
        {
            return 0.5f * (t-a) * (t - a);
        }
        /// <summary>
        /// Partial derivative of the cost with respect to a single output neuron a.
        /// </summary>
        public static float DerivationOfCostFunction(float a, float t)
        {
            return a - t;
        }
    }
}
