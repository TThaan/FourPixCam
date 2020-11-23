using System;

namespace FourPixCam.WeightInits
{
    public class Xavier
    {
        public static float Init(float weight, int n, ActivationType activationType)
        {
            if (activationType == ActivationType.ReLU ||
                activationType == ActivationType.LeakyReLU)
            {
                return ForRelu(weight, n);
            }
            else
            {
                return Standard(weight, n);
            }            
        }
        public static float ForRelu(float weight, int n)
        {
            return weight * (float)Math.Sqrt(2f / n);
        }
        public static float Standard(float weight, int n)
        {
            return weight * (float)Math.Sqrt(1f / n);
        }
    }
}
