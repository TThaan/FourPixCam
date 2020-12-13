using MatrixHelper;
using System;

namespace FourPixCam.Factories
{
    internal static class NeuralNetFactory
    {
        #region fields

        static readonly Random rnd = RandomProvider.GetThreadRandom();
        static NetParameters _netParameters;

        #endregion

        #region internal

        internal static NeuralNet GetNeuralNet(NetParameters netParameters)
        {
            _netParameters = netParameters ?? throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} " +
                    $"({typeof(NeuralNetFactory).Name}.{nameof(GetNeuralNet)})");

            SetWeights(netParameters);
            SetBiases(netParameters);

            return new NeuralNet(netParameters.Layers, netParameters.CostType);
        }
        //internal static NeuralNet GetCopy(this NeuralNet net)
        //{
        //    return new NeuralNet(new Layer[](net.Layers), net.CostType);
        //}

        #endregion

        #region helpers

        static void SetWeights(NetParameters netParameters)
        {
            // Iterate over layers (skip first layer).
            for (int l = 1; l < netParameters.Layers.Length; l++)
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
            for (int l = 1; l < netParameters.Layers.Length; l++)
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

        #endregion
    }
}
