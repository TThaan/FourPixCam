using MatrixHelper;
using System;

namespace FourPixCam
{
    public class NeuralNet
    {
        public int[] NeuronsPerLayer { get; set; }
        public int LayerCount { get; set; }

        public float WeightRange { get; set; }
        public float BiasRange { get; set; }
        public Matrix[] W { get; set; }
        public Matrix[] B { get; set; }
        /// <summary>
        /// input: z, output: a=f(z)
        /// </summary>
        public Func<float,float>[] ActivationDerivations { get; set; }
        /// <summary>
        /// input: z, t: a'=f'(z)=dadz
        /// </summary>
        public Func<float, float, float>[] CostDerivations { get; set; }
    }
}
