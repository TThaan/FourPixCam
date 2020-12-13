using FourPixCam.CostFunctions;
using MatrixHelper;
using NNet_InputProvider;
using System;
using System.Collections.ObjectModel;
using System.Linq;

namespace FourPixCam
{
    /// <summary>
    /// ta Neuron class..(i.e. matrix less variant?)
    /// </summary>
    public class NeuralNet
    {
        #region ctor & fields

        int layerCount;
        Func<Matrix, Matrix, float> costFunction;
        Action<Matrix, Matrix, Matrix> costDerivation;

        // Only needed if ProcessingNet is a child class:
        public NeuralNet(NeuralNet net)
        {

        }
        public NeuralNet(Layer[] layers, CostType costType)
        {
            Layers = layers ??
                throw new NullReferenceException($"{typeof(ObservableCollection<Layer>).Name} {nameof(layers)} " +
                $"({GetType().Name}.ctor)");

            for (int i = 0; i < LayersCount - 1; i++)
            {
                if (i != LayersCount - 1)
                {
                    Layers[i + 1].Processed.ReceptiveField = Layers[i];
                    Layers[i].Processed.ProjectiveField = Layers[i + 1];
                }
            }

            CostType = costType;
        }

        #endregion

        #region internal

        public Layer[] Layers { get; set; }
        public int LayersCount => layerCount == default
            ? layerCount = Layers.Length
            : layerCount;
        public CostType CostType { get; set; }
        // Redundant?:
        public Func<Matrix, Matrix, float> CostFunction => costFunction == default
            ? costFunction = GetCost()
            : costFunction;
        // Redundant?:
        public Action<Matrix, Matrix, Matrix> CostDerivation => costDerivation == default
            ? costDerivation = GetCostDerivation()
            : costDerivation;

        public void FeedForward(Matrix input)
        {
            Layers[0].Processed.ProcessInput(input);
        }
        public void PropagateBack(Matrix expectedOutput, Sample debugSample)
        {
            Layers.Last().Processed.ProcessDelta(expectedOutput, CostDerivation);
        }
        public void AdjustWeightsAndBiases(float learningRate)
        {
            foreach (Layer layer in Layers)
            {
                layer.Processed.AdaptWeightsAndBiases(learningRate);
            }
        }

        #endregion

        #region helpers

        Func<Matrix, Matrix, float> GetCost()
        {
            switch (CostType)
            {
                case CostType.Undefined:
                    return default;
                case CostType.SquaredMeanError:
                    return SquaredMeanError.Cost;
                default:
                    return default;
            }
        }
        Action<Matrix, Matrix, Matrix> GetCostDerivation()
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
