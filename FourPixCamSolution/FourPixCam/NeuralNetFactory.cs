using FourPixCam.Activators;
using MatrixHelper;
using System;

namespace FourPixCam
{
    internal class NeuralNetFactory
    {
        #region ctor & fields

        static readonly Random rnd;

        static NeuralNetFactory()
        {
            rnd = RandomProvider.GetThreadRandom();
        }

        #endregion

        public static NeuralNet GetNeuralNet(string jsonSource)
        {
            // Get from jsonSource later.

            var layers = new[] { 4, 4, 4, 8, 4 };
            float weightRange = 2;
            float biasRange = .1f;

            var result = new NeuralNet()
            {
                NeuronsPerLayer = layers,
                LayerCount = layers.Length,

                WeightRange = weightRange,
                BiasRange = biasRange,

                W = GetWeights(layers, weightRange),
                B = GetBiases(layers, biasRange),
                ActivationDerivations = GetActivations("Implement jsonSource later!"),
            };

            return result;
        }

        #region helpers

        static Matrix[] GetWeights(int[] layers, float weightRange)
        {
            Matrix[] result = new Matrix[layers.Length];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < result.Length; l++)
            {
                Matrix weightsOfThisLayer = new Matrix(layers[l], layers[l - 1]);

                for (int j = 0; j < layers[l]; j++)
                {
                    for (int k = 0; k < layers[l - 1]; k++)
                    {
                        // The entry in the j-th row and k-th colum is w^l_jk
                        // i.e. the weight connecting
                        // from the k-th neuron of layer l-1
                        // to the jth neuron of layer l.
                        weightsOfThisLayer[j, k] = weightRange / 2 * GetRandom10th();// * GetSmallRandomNumber();
                    }
                };

                result[l] = weightsOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        static Matrix[] GetBiases(int[] layers, float biasRange)
        {
            Matrix[] result = new Matrix[layers.Length];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < result.Length; l++)
            {
                Matrix biasesOfThisLayer = new Matrix(layers[l]);

                for (int j = 0; j < layers[l]; j++)
                {
                    biasesOfThisLayer[j] = biasRange / 2;
                };

                result[l] = biasesOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        /// <summary>
        /// Better in RandomProvider?
        /// </summary>
        static float GetSmallRandomNumber()
        {
            return (float)(.0009 * rnd.NextDouble() + .0001) * (rnd.Next(2) == 0 ? -1 : 1);
        }
        static Func<float, float>[] GetActivations(string jsonSource)
        {

            // Get from jsonSource later.

            return new Func<float, float>[]
            {
                default,   // Skip activator for first "layer".
                Sigmoid.a,
                Sigmoid.a,
                ReLU.a, // Try LeakyReLU here.
                ReLU.a
            };
        }
        static float GetRandom10th()
        {
            var x = (rnd.NextDouble() + .1f);
            return (float)Math.Round(x <= .9 ? x : .9, 1);
        }

        #endregion

    }
}
