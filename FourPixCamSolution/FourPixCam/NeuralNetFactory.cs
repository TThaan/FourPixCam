using FourPixCam.Activators;
using FourPixCam.CostFunctions;
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
            int weightMin = -1;
            int weightMax = 1;
            int biasMin = -2;
            int biasMax = 2;

            var result = new NeuralNet()
            {
                NeuronsPerLayer = layers,
                LayerCount = layers.Length,
                WeightMin = weightMin,
                WeightMax = weightMax,
                BiasMin = biasMin,
                BiasMax = biasMax,
                W = GetWeights(layers, weightMin, weightMax),
                B = GetBiases(layers, biasMin, biasMax),
                Activations = GetActivations("Implement jsonSource later!"),
                ActivationDerivations = GetActivationDerivations("Implement jsonSource later!"),
                CostDerivation = GetCostDerivation("Implement jsonSource later!")
            };

            return result;
        }

        #region helpers

        static Matrix[] GetWeights(int[] layers, int weightMin, int weightMax)
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
                        weightsOfThisLayer[j, k] = GetRandom10th(weightMin + (weightMax - weightMin) * rnd.NextDouble());// * GetSmallRandomNumber();
                    }
                };

                result[l] = weightsOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        static Matrix[] GetBiases(int[] layers, int biasMin, int biasMax)
        {
            Matrix[] result = new Matrix[layers.Length];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < result.Length; l++)
            {
                Matrix biasesOfThisLayer = new Matrix(layers[l]);

                for (int j = 0; j < layers[l]; j++)
                {
                    var x = GetRandom10th(biasMin + (biasMax - biasMin) * rnd.NextDouble());
                    biasesOfThisLayer[j] = x;
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
        static Func<float, float>[] GetActivationDerivations(string jsonSource)
        {
            // Get from jsonSource later.

            return new Func<float, float>[]
            {
                default,   // Skip activator for first "layer".
                Sigmoid.dadz,
                Sigmoid.dadz,
                ReLU.dadz, // Try LeakyReLU here.
                ReLU.dadz
            };
        }
        static Func<float, float, float> GetCostDerivation(string jsonSource)
        {
            // Get from jsonSource later.

            return  SquaredMeanError.DerivationOfCostFunction;
        }
        static float GetRandom10th(double x)
        {
            double ratio = (rnd.NextDouble() + .1f);
            //var y = (float)Math.Round(ratio <= .9 ? ratio : .9, 1);

            return (float)Math.Round(ratio * x, 1);    //rnd.Next(0,2) == 0 ? y : -y
        }

        #endregion

    }
}
