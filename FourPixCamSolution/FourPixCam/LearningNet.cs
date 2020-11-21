﻿using FourPixCam.CostFunctions;
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

        #endregion

        #region ctor

        public LearningNet(NeuralNet net)      // param jasonFile
        {
            this.Net = net;

            // CostType = CostType.SquaredMeanError;

            Z = new Matrix[net.LayerCount];
            A = new Matrix[net.LayerCount];
            dadz_OfLayer = new Matrix[net.LayerCount];
            Delta = new Matrix[net.LayerCount];
        }

        #region helper methods

        #endregion

        #endregion

        #region properties

        public NeuralNet Net { get; }
        // public Matrix x { get; set; }

        /// <summary>
        /// expected output
        /// </summary>
        //public Matrix t { get; set; }   // redundant?
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

        //public CostType CostType { get; set; }
        public float LastCost { get; set; } // redundant?


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
                    Net.W[i].DumpToConsole($"\nW{i} = "),
                    A[i - 1], 
                    Net.B[i].DumpToConsole($"\nB{i} = "));
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
                //dadz_OfLayer[l] = NeurNetMath.Get_dadz(A[l], Z[l], net.CostDerivation)
                //    .DumpToConsole($"\ndadz{l} =");
                Matrix delta;

                if (l == Net.LayerCount - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
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
