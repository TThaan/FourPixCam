using System.Collections.Generic;

namespace FourPixCam
{
    /// <summary>
    /// Model for UI
    /// </summary>
    public class NetParameters
    {
        #region ctor

        public NetParameters()
        {
            // Default Values:

            IsWithBias = false;
            WeightMin = -1;
            WeightMax = 1;
            BiasMin = -1;
            BiasMax = 1;
            Layers = new List<Layer>()
            {
                new Layer{ N=4, ActivationType=ActivationType.None},
                new Layer{ N=4, ActivationType=ActivationType.Tanh},
                new Layer{ N=4, ActivationType=ActivationType.Tanh},
                new Layer{ N=8, ActivationType=ActivationType.ReLU},
                new Layer{ N=4, ActivationType=ActivationType.Tanh}
            };
            CostType = CostType.SquaredMeanError;
            WeightInitType = WeightInitType.Xavier;
        }

        #endregion

        #region public

        public List<Layer> Layers { get; set; }
        public bool IsWithBias { get; set; }
        public float WeightMin { get; set; }
        public float WeightMax { get; set; }
        public float BiasMin { get; set; }
        public float BiasMax { get; set; }
        public CostType CostType { get; set; }
        public WeightInitType WeightInitType { get; set; }

        #endregion
    }
}
