using MatrixHelper;
using System;

namespace FourPixCam
{
    public class NeuralNet
    {
        public int[] NeuronsPerLayer { get; set; }
        public int LayerCount { get; set; }
        public Matrix[] W { get; set; }
        public Matrix[] B { get; set; }
        public Func<float, float>[] Activations { get; set; }
        public Func<float,float>[] ActivationDerivations { get; set; }
        public Func<float, float, float> CostDerivation { get; set; }
        public bool IsWithBias { get; set; }
    }
}
