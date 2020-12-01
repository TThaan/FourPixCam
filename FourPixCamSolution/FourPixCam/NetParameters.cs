using FourPixCam.WeightInits;
using System;

namespace FourPixCam
{
    /// <summary>
    /// Model for UI
    /// </summary>
    [Serializable]
    public class NetParameters
    {
        #region fields

        Func<float, int, ActivationType, float> weightInit;

        #endregion

        #region ctor

        public NetParameters()
        {
            // Default Values (Redundant, only wanted in/for UI?):
            // SetDefaultValues();
        }

        #region helpers

        void SetDefaultValues()
        {
            IsWithBias = false;
            WeightMin = -1;
            WeightMax = 1;
            BiasMin = -1;
            BiasMax = 1;
            Layers = new[]
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

        #endregion

        #region public

        public Layer[] Layers { get; set; }
        public bool IsWithBias { get; set; }
        public float WeightMin { get; set; }
        public float WeightMax { get; set; }
        public float BiasMin { get; set; }
        public float BiasMax { get; set; }
        public CostType CostType { get; set; }
        public WeightInitType WeightInitType { get; set; }

        public Func<float, int, ActivationType, float> WeightInit => weightInit == default
            ? weightInit = GetWeightInit()
            : weightInit;

        // Actually not NetParameters but rather TrainingParameters..

        public float LearningRate { get; set; }
        public float ChangeOfLearningRate { get; set; }
        public int EpochCount { get; set; }

        #endregion

        #region helpers

        // better in factory?
        Func<float, int, ActivationType, float> GetWeightInit()
        {
            switch (WeightInitType)
            {
                case WeightInitType.None:
                    return default;
                case WeightInitType.Xavier:
                    return Xavier.Init;
                default:
                    return default;
            }
        }

        #endregion
    }
}
