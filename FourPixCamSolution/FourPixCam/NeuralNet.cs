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
    public class NeuralNet
    {
        #region ctor & fields

        int layerCount;
        Func<Matrix, Matrix, Matrix> cost, costDerivation; // Redundant? (Not saving values here but refs to methods)

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
            // Layers[Layers.Length - 2].Processed.ProjectiveField = Layers[Layers.Length - 1];

            CostType = costType;

            // Define cost and costDerivation here?!
        }

        #endregion

        #region internal

        public Layer[] Layers { get; set; }
        public int LayersCount => layerCount == default
            ? layerCount = Layers.Length
            : layerCount;
        public CostType CostType { get; set; }
        // Redundant?:
        public Func<Matrix, Matrix, Matrix> Cost => cost == default
            ? cost = GetCost()
            : cost;
        // Redundant?:
        public Func<Matrix, Matrix, Matrix> CostDerivation => costDerivation == default
            ? costDerivation = GetCostDerivation()
            : costDerivation;

        public void FeedForward(Matrix input)
        {
            Layers[0].Processed.ProcessInput(input);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>cost matrix?</returns>
        public void PropagateBack(Matrix expectedOutput, float learningRate)
        {
            Layers.Last().Processed.ProcessOutputDelta(expectedOutput, CostDerivation, learningRate);
        }

        #endregion

        #region helpers

        Func<Matrix, Matrix, Matrix> GetCost()
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
        Func<Matrix, Matrix, Matrix> GetCostDerivation()
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
