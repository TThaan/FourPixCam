using FourPixCam.CostFunctions;
using MatrixHelper;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    public class LearningNet
    {
        #region ctor

        public LearningNet(NeuralNet net)      // param jasonFile
        {
            Net = net;
            Z = new Matrix[net.LayerCount];
            A = new Matrix[net.LayerCount];
            dadz_OfLayer = new Matrix[net.LayerCount];
            Delta = new Matrix[net.LayerCount];
        }

        #endregion

        #region properties

        public NeuralNet Net { get; }
        public Matrix[] Z { get; set; }
        public Matrix[] A { get; set; }
        public Matrix[] dadz_OfLayer { get; set; }
        public Matrix[] Delta { get; set; }

        #endregion

        #region methods

        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            // wa: Separate inp layer from 'layers' ?!
            A[0] = input;

            // iterate over layers (skip input layer)
            for (int i = 1; i < Net.LayerCount; i++)
            {
                Z[i] = Get_z(
                    Net.W[i].Log($"\nW{i} = "),
                    A[i - 1], 
                    Net.B[i].Log($"\nB{i} = "));
                A[i] = Get_a(Z[i], Net.Activations[i]);
            }

            return A.Last();
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

                if (l == Net.LayerCount - 1)
                {
                    delta = Get_deltaOutput(A[l], t, Net.CostDerivation, Z[l], Net.ActivationDerivations[l]);
                }
                else
                {
                    delta = Get_deltaHidden(Net.W[l + 1], Delta[l + 1], Z[l], Net.ActivationDerivations[l]);
                }

                Delta[l] = delta;
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
