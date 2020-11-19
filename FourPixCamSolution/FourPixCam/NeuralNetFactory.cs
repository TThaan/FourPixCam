using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using FourPixCam.WeightInits;
using MatrixHelper;
using System;

namespace FourPixCam
{
    internal static class NeuralNetFactory
    {
        #region ctor & fields

        static readonly Random rnd;

        static NeuralNetFactory()
        {
            rnd = RandomProvider.GetThreadRandom();
        }

        #endregion

        public static NeuralNet GetNeuralNet(string jsonSource, bool isWithBias)
        {
            // Get from jsonSource later.

            var layers = new[] { 4, 4, 4, 8, 4 };
            float weightMin = -1f;
            float weightMax = 1f;
            float biasMin = -1f;
            float biasMax = 1f;
            Func<float, float>[] activations = GetActivations("Implement jsonSource later!");
            var w = GetWeights(layers, weightMin, weightMax);
            Func<float, int, Type, float> weightInit = Xavier.Init;
            AdaptWeights(w, weightInit, activations);
            var b = isWithBias ? GetBiases(layers, biasMin, biasMax) : GetBiases(layers, 0, 0);

            var result = new NeuralNet()
            {
                NeuronsPerLayer = layers,
                LayerCount = layers.Length,
                WeightMin = weightMin,
                WeightMax = weightMax,
                BiasMin = biasMin,
                BiasMax = biasMax,
                W = w,
                B = b,
                Activations = activations,
                ActivationDerivations = GetActivationDerivations("Implement jsonSource later!"),
                CostDerivation = GetCostDerivation("Implement jsonSource later!"),
                IsWithBias = isWithBias
            };

            return result;
        }

        public static NeuralNet GetCopy(this NeuralNet net)
        {
            return new NeuralNet()
            {
                NeuronsPerLayer = net.NeuronsPerLayer,
                LayerCount = net.LayerCount,
                WeightMin = net.WeightMin,
                WeightMax = net.WeightMax,
                BiasMin = net.BiasMin,
                BiasMax = net.BiasMax,
                W = net.W,
                B = net.B,
                Activations = net.Activations,
                ActivationDerivations = net.ActivationDerivations,
                CostDerivation = net.CostDerivation,
                IsWithBias = net.IsWithBias
            };
        }

        #region helpers

        static Matrix[] GetWeights(int[] layers, float weightMin, float weightMax)
        {
            Matrix[] result = new Matrix[layers.Length];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < result.Length; l++)
            {
                Matrix w = new Matrix(layers[l], layers[l - 1]);

                for (int j = 0; j < w.m; j++)
                {
                    for (int k = 0; k < w.n; k++)
                    {
                        w[j, k] = (float)((weightMin + (weightMax - weightMin) * rnd.NextDouble()));// * GetSmallRandomNumber();
                    }
                };

                result[l] = w;   // wa: result[0]?
            }

            return result;
        }
        static Matrix[] GetBiases(int[] layers, float biasMin, float biasMax)
        {
            Matrix[] result = new Matrix[layers.Length];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < result.Length; l++)
            {
                Matrix biasesOfThisLayer = new Matrix(layers[l]);

                for (int j = 0; j < layers[l]; j++)
                {
                    var x = GetRandom10th(biasMin + (biasMax - biasMin) * rnd.NextDouble());
                    biasesOfThisLayer[j] = x;//0
                };

                result[l] = biasesOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        static void AdaptWeights(Matrix[] weightMatrices, Func<float, int, Type, float> weightInit, Func<float, float>[] activations)
        {
            for (int l = 1; l < weightMatrices.Length; l++)
            {
                Matrix w = weightMatrices[l];

                for (int j = 0; j < w.m; j++)
                {
                    for (int k = 0; k < w.n; k++)
                    {
                        float oldW = w[j, k];
                        w[j, k] = weightInit(w[j, k], w.n, activations[l].Method.DeclaringType);
                        float newW = w[j, k];
                    }
                };
            }
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
                Tanh.a,
                Tanh.a,
                ReLU.a, // Try LeakyReLU here.
                Tanh.a
            };
        }
        static Func<float, float>[] GetActivationDerivations(string jsonSource)
        {
            // Get from jsonSource later.

            return new Func<float, float>[]
            {
                default,   // Skip activator for first "layer".
                Tanh.dadz,
                Tanh.dadz,
                ReLU.dadz, // Try LeakyReLU here.
                Tanh.dadz
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

            return (float)Math.Round(ratio * x, 4);    //rnd.Next(0,2) == 0 ? y : -y
        }

        #endregion
    }
}
