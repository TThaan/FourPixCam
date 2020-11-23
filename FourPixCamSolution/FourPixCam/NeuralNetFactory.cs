using MatrixHelper;
using System;
using System.Collections.ObjectModel;

namespace FourPixCam
{
    internal static class NeuralNetFactory
    {
        #region fields

        static readonly Random rnd = RandomProvider.GetThreadRandom();
        static NetParameters _netParameters;

        #endregion

        #region public

        public static NeuralNet GetNeuralNet(NetParameters netParameters)
        {
            _netParameters = netParameters ?? throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} " +
                    $"({typeof(NeuralNetFactory).Name}.{nameof(GetNeuralNet)})");

            SetWeights(netParameters);
            SetBiases(netParameters);

            return new NeuralNet(netParameters.Layers, netParameters.CostType);
        }
        public static NeuralNet GetCopy(this NeuralNet net)
        {
            return new NeuralNet(new ObservableCollection<Layer>(net.Layers), net.CostType);
        }

        #endregion

        #region helpers

        static void SetWeights(NetParameters netParameters)
        {
            // Iterate over layers (skip first layer).
            for (int l = 1; l < netParameters.Layers.Count; l++)
            {
                Matrix weights = new Matrix(
                    netParameters.Layers[l].N, 
                    netParameters.Layers[l - 1].N);

                for (int j = 0; j < weights.m; j++)
                {
                    for (int k = 0; k < weights.n; k++)
                    {
                        float baseValue = (float)(netParameters.WeightMin + 
                            (netParameters.WeightMax - netParameters.WeightMin) * rnd.NextDouble());
                        weights[j, k] = netParameters.WeightInit(baseValue, weights.n, netParameters.Layers[l].ActivationType);
                    }
                };

                netParameters.Layers[l].Weights = weights;
            }
        }
        static void SetBiases(NetParameters netParameters)
        {
            if (!netParameters.IsWithBias)
                return;

            // Iterate over layers (skip first layer).
            for (int l = 1; l < netParameters.Layers.Count; l++)
            {
                Matrix biases = new Matrix(netParameters.Layers[l].N);

                for (int j = 0; j < netParameters.Layers[l].N; j++)
                {
                    biases[j] = (float)(netParameters.BiasMin + 
                        (netParameters.BiasMax - netParameters.BiasMin) * rnd.NextDouble());
                };

                netParameters.Layers[l].Biases = biases;
            }
        }

        //static Matrix[] GetWeights(int[] layers, float weightMin, float weightMax, Func<float, int, Type, float> weightInit, Func<float, float>[] activations)
        //{
        //    Matrix[] result = new Matrix[layers.Length];

        //    // Iterate over layers (skip first layer).
        //    for (int l = 1; l < result.Length; l++)
        //    {
        //        Matrix w = new Matrix(layers[l], layers[l - 1]);

        //        for (int j = 0; j < w.m; j++)
        //        {
        //            for (int k = 0; k < w.n; k++)
        //            {
        //                float baseValue = (float)((weightMin + (weightMax - weightMin) * rnd.NextDouble()));
        //                w[j, k] = weightInit(baseValue, w.n, activations[l].Method.DeclaringType);
        //            }
        //        };

        //        result[l] = w;
        //    }

        //    return result;
        //}
        //static Matrix[] GetBiases(int[] layers, float biasMin, float biasMax)
        //{
        //    Matrix[] result = new Matrix[layers.Length];

        //    // Iterate over layers (skip first layer).
        //    for (int l = 1; l < result.Length; l++)
        //    {
        //        Matrix biasesOfThisLayer = new Matrix(layers[l]);

        //        for (int j = 0; j < layers[l]; j++)
        //        {
        //            biasesOfThisLayer[j] = (float)(biasMin + (biasMax - biasMin) * rnd.NextDouble());
        //        };

        //        result[l] = biasesOfThisLayer;
        //    }

        //    return result;
        //}


        //static Func<float, float>[] GetActivations(string jsonSource)
        //{
        //    // Get from jsonSource later.

        //    return new Func<float, float>[]
        //    {
        //        default,   // Skip activator for first "layer".
        //        Tanh.a,
        //        Tanh.a,
        //        ReLU.a,
        //        Tanh.a
        //    };
        //}

        //static Func<float, float>[] GetActivationDerivations(string jsonSource)
        //{
        //    // Get from jsonSource later.. no, conclude from activations!

        //    return new Func<float, float>[]
        //    {
        //        default,   // Skip activator for first "layer".
        //        Tanh.dadz,
        //        Tanh.dadz,
        //        ReLU.dadz,
        //        Tanh.dadz
        //    };
        //}
        //static Func<float, float, float> GetCostDerivation(string jsonSource)
        //{
        //    // Get from jsonSource later.

        //    return SquaredMeanError.DerivationOfCostFunction;
        //}

        #endregion
    }
}
