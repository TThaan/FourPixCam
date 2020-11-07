using FourPixCam.Activators;
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

            Z = new Matrix[net.L];
            A = new Matrix[net.L];
            dadz_OfLayer = new Matrix[net.L];
            E = new Matrix[net.L];
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
        public Matrix[] E { get; set; }
        public Matrix[] F { get; set; } // => activations[]

        public CostType CostType { get; set; }
        public float LastCost { get; set; } // redundant?


        #endregion

        #region methods

        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine($"                                        F E E D   F O R W A R D");
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine();

            // wa: Separate inp layer from 'layers' ?!
            A[0] = input.DumpToConsole($"\nA[0] = "); //new Matrix(input.ToArray());
            Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");


            // iterate over layers (skip input layer)
            for (int i = 1; i < net.L; i++)
            {
                Z[i] = NeurNetMath.z(net.W[i].DumpToConsole($"\nW{i} = "), A[i - 1].DumpToConsole($"\nA{i-1} = "), net.B[i].DumpToConsole($"\nB{i} = ")).DumpToConsole($"\nZ{i} = ");
                A[i] = NeurNetMath.a(Z[i], net.ActivationTypes[i]).DumpToConsole($"\nA{i} = ");
                Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");
            }

            return A.Last();
        }
        public void BackPropagate(Matrix y, float learningRate)
        {
            // debug
            // var v1 = NeurNetMath.C(a_OfLayer[net.L-1], y, CostType.SquaredMeanError);

            // Iterate backwards over each layer (skip input layer).
            for (int l = net.L - 1; l > 0; l--)
            {
                dadz_OfLayer[l] = NeurNetMath.dadz(Z[l], net.ActivationTypes[l]);
                Matrix error;

                if (l == net.L - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    error = NeurNetMath.deltaOfOutputLayer(A[l], y, CostType.SquaredMeanError, dadz_OfLayer[l]);
                }
                else
                {
                    error = NeurNetMath.deltaOfHiddenLayer(net.W[l + 1], E[l + 1], dadz_OfLayer[l]);
                }

                E[l] = error;
            }

            // Adjust weights and biases.
            for (int l = 0; l < net.L - 1; l++)
            {
                net.W[l] = GetCorrectedMatrix(net.W[l], A[l-1], E[l], learningRate);
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
