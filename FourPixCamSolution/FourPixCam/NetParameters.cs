using System.Collections.Generic;
using System.Collections.ObjectModel;

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
            Layers = new ObservableCollection<Layer>()
            {
                new Layer{ Id= 0, N=4, ActivationType=ActivationType.None},
                new Layer{ Id= 1, N=4, ActivationType=ActivationType.Tanh},
                new Layer{ Id= 2, N=4, ActivationType=ActivationType.Tanh},
                new Layer{ Id= 3, N=8, ActivationType=ActivationType.ReLU},
                new Layer{ Id= 4, N=4, ActivationType=ActivationType.Tanh}
            };
            CostType = CostType.SquaredMeanError;
            WeightInitType = WeightInitType.Xavier;
        }

        #endregion

        #region public

        public ObservableCollection<Layer> Layers { get; set; }
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
