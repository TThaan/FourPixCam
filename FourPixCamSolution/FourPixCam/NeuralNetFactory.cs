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

        static readonly Random rnd = RandomProvider.GetThreadRandom();

        #endregion

        #region public

        public static NeuralNet GetNeuralNet(string jsonSource, bool isWithBias)
        {
            // Get from jsonSource later.

            var layers = new[] { 4, 4, 4, 8, 4 };
            float 
                weightMin = -1f,
                weightMax = 1f,
                biasMin = -1f,
                biasMax = 1f;

            Func<float, float>[] activations = GetActivations("Implement jsonSource later!");
            var w = GetWeights(layers, weightMin, weightMax, Xavier.Init, activations);
            var b = isWithBias ? GetBiases(layers, biasMin, biasMax) : GetBiases(layers, 0, 0);

            var result = new NeuralNet()
            {
                NeuronsPerLayer = layers,
                LayerCount = layers.Length,
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
                W = net.W,
                B = net.B,
                Activations = net.Activations,
                ActivationDerivations = net.ActivationDerivations,
                CostDerivation = net.CostDerivation,
                IsWithBias = net.IsWithBias
            };
        }

        #endregion

        #region helpers

        static Func<float, float>[] GetActivations(string jsonSource)
        {
            // Get from jsonSource later.

            return new Func<float, float>[]
            {
                default,   // Skip activator for first "layer".
                Tanh.a,
                Tanh.a,
                ReLU.a,
                Tanh.a
            };
        }
        static Func<float, float>[] GetActivationDerivations(string jsonSource)
        {
            // Get from jsonSource later.. no, conclude from activations!

            return new Func<float, float>[]
            {
                default,   // Skip activator for first "layer".
                Tanh.dadz,
                Tanh.dadz,
                ReLU.dadz,
                Tanh.dadz
            };
        }
        static Func<float, float, float> GetCostDerivation(string jsonSource)
        {
            // Get from jsonSource later.

            return SquaredMeanError.DerivationOfCostFunction;
        }
        static Matrix[] GetWeights(int[] layers, float weightMin, float weightMax, Func<float, int, Type, float> weightInit, Func<float, float>[] activations)
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
                        float baseValue = (float)((weightMin + (weightMax - weightMin) * rnd.NextDouble()));
                        w[j, k] = weightInit(baseValue, w.n, activations[l].Method.DeclaringType);
                    }
                };

                result[l] = w;
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
                    biasesOfThisLayer[j] = (float)(biasMin + (biasMax - biasMin) * rnd.NextDouble());
                };

                result[l] = biasesOfThisLayer;
            }

            return result;
        }

        #endregion
    }
}
