using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    public class LearningNet
    {
        #region fields

        // readonly Random rnd = RandomProvider.GetThreadRandom();        
        NeuralNet net;

        #endregion

        #region ctor

        public LearningNet(NeuralNet net)      // param jasonFile
        {
            this.net = net;

            CostType = CostType.SquaredMeanError;

            Z = new Matrix[net.LayerCount];
            A = new Matrix[net.LayerCount];
            dadz_OfLayer = new Matrix[net.LayerCount];
            Delta = new Matrix[net.LayerCount];
        }

        #region helper methods

        #endregion

        #endregion

        #region properties

        // public Matrix x { get; set; }

        /// <summary>
        /// expected output
        /// </summary>
        public Matrix t { get; set; }   // redundant?
        public float C { get; set; }
        /// <summary>
        /// total value (= wa + b)
        /// </summary>
        public Matrix[] Z { get; set; }
        /// <summary>
        /// activation (= f(z))
        /// </summary>
        public Matrix[] A { get; set; }
        public Matrix[] dadz_OfLayer { get; set; }  // => Matrix.Partial(f, a);
        public Matrix[] Delta { get; set; }
        public Matrix[] F { get; set; } // => activations[]

        public CostType CostType { get; set; }
        public float LastCost { get; set; } // redundant?


        #endregion

        #region methods

        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            $"F E E D   F O R W A R D".WriteDumpingTitle();

            // wa: Separate inp layer from 'layers' ?!
            A[0] = input.DumpToConsole($"\nA[0] = "); //new Matrix(input.ToArray());
            Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");


            // iterate over layers (skip input layer)
            for (int i = 1; i < net.LayerCount; i++)
            {
                Z[i] = NeurNetMath.Get_z(net.W[i].DumpToConsole($"\nW{i} = "), A[i - 1].DumpToConsole($"\nA{i-1} = "), net.B[i].DumpToConsole($"\nB{i} = ")).DumpToConsole($"\nZ{i} = ");
                A[i] = NeurNetMath.Get_a(Z[i], net.ActivationDerivations[i]).DumpToConsole($"\nA{i} = ");
                Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");
            }

            return A.Last();
        }
        public void BackPropagate(Matrix y, float learningRate)
        {
            $"B A C K P R O P A P A G A T I O N".WriteDumpingTitle();

            // debug
            var c = NeurNetMath.Get_C(A[net.LayerCount - 1], y, SquaredMeanError.CostFunction)
                .DumpToConsole($"\n{CostType.SquaredMeanError} C =");
            var cTotal = NeurNetMath.Get_CTotal(A[net.LayerCount - 1], y, SquaredMeanError.CostFunction);
            Console.WriteLine($"\nCTotal = {cTotal}\n");

            Matrix[] nextW = new Matrix[net.LayerCount];
            Matrix[] nextB = new Matrix[net.LayerCount];

            // Iterate backwards over each layer (skip input layer).
            for (int l = net.LayerCount - 1; l > 0; l--)
            {
                dadz_OfLayer[l] = NeurNetMath.Get_dadz(A[l], Z[l], net.CostDerivations[l])
                    .DumpToConsole($"\ndadz{l} =");
                Matrix delta;

                if (l == net.LayerCount - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    delta = NeurNetMath.Get_deltaOutput(A[l], t, net.CostDerivations[l], Z[l], net.ActivationDerivations[l]); //
                }
                else
                {
                    delta = NeurNetMath.Get_deltaHidden(net.W[l + 1], Delta[l + 1], Z[l], net.ActivationDerivations[l]);//, A[l]
                }

                Delta[l] = delta.DumpToConsole($"\ndelta{l} =");
                nextW[l] = Get_CorrectedWeights(net.W[l], A[l - 1], Delta[l], learningRate)
                    .DumpToConsole($"\nnextW{l} =");
                nextB[l] = Get_CorrectedWeights(net.B[l], A[l - 1], Delta[l], learningRate)
                    .DumpToConsole($"\nnextB{l} =");
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
