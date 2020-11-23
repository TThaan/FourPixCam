using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    public class LearningNet
    {
        #region ctor & fields

        Layer[] _layers;

        public LearningNet(NeuralNet net)
        {
            Net = net ??
                throw new NullReferenceException($"{typeof(NeuralNet).Name} {nameof(net)} " +
                $"({GetType().Name}.ctor)");
            _layers = net.Layers.ToArray();
        }

        #endregion

        #region properties

        public NeuralNet Net { get; }
        // public Matrix[] Z { get; set; }
        // public Matrix[] A { get; set; }
        // public Matrix[] Delta { get; set; }

        #endregion

        #region methods

        // redundant?
        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            return Net.FeedForward(input);
        }
        public void BackPropagate(Matrix t, float learningRate)
        {
            // debug
            var c = Get_C(A[Net.LayerCount - 1], t, SquaredMeanError.CostFunction);
            var cTotal = Get_CTotal(A[Net.LayerCount - 1], t, SquaredMeanError.CostFunction);

            Matrix[] nextW = new Matrix[Net.LayerCount];
            Matrix[] nextB = new Matrix[Net.LayerCount];

            // Iterate backwards over each layer (skip input layer).
            for (int l = Net.LayerCount - 1; l > 0; l--)
            {
                Matrix delta;

                nextW[l] = Get_CorrectedWeights(Net.W[l], Delta[l], A[l - 1], learningRate);
                if (Net.IsWithBias)
                {
                    nextB[l] = Get_CorrectedBiases(Net.B[l], Delta[l], learningRate);
                }
            }

            Net.W = nextW;
            if (Net.IsWithBias)
            {
                Net.B = nextB;
            }
        }

        #endregion
    }
}
