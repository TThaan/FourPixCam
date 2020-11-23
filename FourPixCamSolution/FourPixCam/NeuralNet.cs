using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    /// <summary>
    /// ta Neuron class..(i.e. matrix less variant?)
    /// </summary>
    public class NeuralNet
    {
        #region ctor & fields

        int layerCount;
        Func<float, float, float> cost, costDerivation; // Redundant? (Not saving values here but refs to methods!)

        public NeuralNet(ObservableCollection<Layer> layers, CostType costType)
        {
            Layers = layers ??
                throw new NullReferenceException($"{typeof(ObservableCollection<Layer>).Name} {nameof(layers)} " +
                $"({GetType().Name}.ctor)");

            // Define cost and costDerivation here?!
        }

        #endregion

        #region public

        public ObservableCollection<Layer> Layers { get; set; }
        public int LayerCount => layerCount = default
            ? layerCount = Layers.Count
            : layerCount;
        public CostType CostType { get; set; }
        // Redundant? (Not saving values here but refs to methods!):
        public Func<float, float, float> Cost => cost = default
            ? cost = GetCost()
            : cost;
        // Redundant? (Not saving values here but refs to methods!):
        public Func<float, float, float> CostDerivation => costDerivation = default
            ? costDerivation = GetCostDerivation()
            : costDerivation;

        public Matrix FeedForward(Matrix input)
        {
            //Layers[0].Processed.Input = input;

            //foreach (var layer in Layers)
            //{
            //    if (layer.Id > 0)
            //    {
            //        layer.Processed.Input = Get_z(layer.Weights, 
            //            Layers.Single(x => x.Id == layer.Id - 1).Processed.Input, // use ReceptiveField?
            //            layer.Biases);
            //    }
            //    layer.Processed.Output = Get_a(layer.Processed.Input, layer.Activation);
            //}

            Layers[0].Processed.ProcessInput(input);
            return Layers.Last().Processed.Output;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>cost matrix?</returns>
        public void PropagateBack(Matrix expectedOutput)
        {
            Layers.Last().Processed.ProcessCost(expectedOutput, CostDerivation);
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
