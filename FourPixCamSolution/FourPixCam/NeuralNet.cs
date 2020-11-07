using FourPixCam.Enums;
using MatrixHelper;

namespace FourPixCam
{
    public class NeuralNet
    {
        public int[] NeuronsPerLayer { get; set; }
        public int L { get; set; }

        public float WeightRange { get; set; }
        public float BiasRange { get; set; }
        public Matrix[] W { get; set; }
        public Matrix[] B { get; set; }
        public ActivationType[] ActivationTypes { get; set; }
    }
}
