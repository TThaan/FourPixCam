using MatrixHelper;

namespace _4PixCam.CostFunctions
{
    class SquaredMeanError : CostFunction
    {
        public override float GetCost(float a, float t)
        {
            return 0.5f * (t-a) * (t - a);
        }
        public override Matrix GetCost(Matrix a, Matrix t)
        {
            return 0.5f * (t - a) * (t - a);
        }
        public override float GetCostDerivative(float a, float t)
        {
            return a - t;
        }
        public override Matrix GetCostDerivative(Matrix a, Matrix t)
        {
            return a - t;
        }
    }
}
