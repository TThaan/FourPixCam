using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Collections.ObjectModel;
using System.Linq;

namespace FourPixCam
{
    /// <summary>
    /// ta Neuron class..(i.e. matrix less variant?)
    /// </summary>
    internal class NeuralNet
    {
        #region ctor & fields

        int layerCount;
        Func<float, float, float> cost, costDerivation; // Redundant? (Not saving values here but refs to methods!)

        // Only needed if ProcessingNet is a child class:
        internal NeuralNet(NeuralNet net)
        {

        }
        internal NeuralNet(Layer[] layers, CostType costType)
        {
            Layers = layers ??
                throw new NullReferenceException($"{typeof(ObservableCollection<Layer>).Name} {nameof(layers)} " +
                $"({GetType().Name}.ctor)");

            for (int i = 1; i < LayersCount; i++)
            {
                Layers[i].Processed.ReceptiveField = Layers[i - 1];
                Layers[i-1].Processed.ProjectiveField = Layers[i];
            }

            CostType = costType;

            // Define cost and costDerivation here?!
        }

        #endregion

        #region internal

        internal Layer[] Layers { get; set; }
        internal int LayersCount => layerCount == default
            ? layerCount = Layers.Length
            : layerCount;
        internal CostType CostType { get; set; }
        // Redundant? (Not saving values here but refs to methods!):
        internal Func<float, float, float> Cost => cost == default
            ? cost = GetCost()
            : cost;
        // Redundant? (Not saving values here but refs to methods!):
        internal Func<float, float, float> CostDerivation => costDerivation == default
            ? costDerivation = GetCostDerivation()
            : costDerivation;

        internal Matrix FeedForward(Matrix input)
        {
            Layers[0].Processed.ProcessInput(input);
            return Layers.Last().Processed.Output;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>cost matrix?</returns>
        internal void PropagateBack(Matrix expectedOutput, float learningRate)
        {
            Layers.Last().Processed.ProcessCost(expectedOutput, CostDerivation, learningRate);
        }
        internal void AdaptWeightsAndBiases(float learningRate)
        {
            
        }

        #endregion

        #region helpers

        Func<float, float, float> GetCost()
        {
            switch (CostType)
            {
                case CostType.Undefined:
                    return default;
                case CostType.SquaredMeanError:
                    return SquaredMeanError.CostFunction;
                default:
                    return default;
            }
        }
        Func<float, float, float> GetCostDerivation()
        {
            switch (CostType)
            {
                case CostType.Undefined:
                    return default;
                case CostType.SquaredMeanError:
                    return SquaredMeanError.DerivationOfCostFunction;
                default:
                    return default;
            }
        }

        #endregion
    }
}
