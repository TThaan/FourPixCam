using MatrixHelper;

namespace _4PixCam.CostFunctions
{
    /// <summary>
    /// Better return array (= matrices)?
    /// </summary>
    public abstract class CostFunction
    {
        /// <summary>
        /// Get cost of a single trainings sample.
        /// </summary>
        /// <param name="a">output</param>
        /// <param name="t">t = expected output</param>
        /// <returns></returns>
        public abstract float GetCost(float a, float t);
        public abstract Matrix GetCost(Matrix a, Matrix t);
        /// <summary>
        /// Get derivative of the cost 
        /// with respect to the activation function of the output neuron.
        /// </summary>
        public abstract float GetCostDerivative(float a, float t);
        public abstract Matrix GetCostDerivative(Matrix a, Matrix t);
    }
}
