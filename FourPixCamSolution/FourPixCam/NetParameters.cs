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
        #region ctor

        public NetParameters(WeightInitType weightInitType)
        {
            WeightInitType = weightInitType;
            WeightInit = GetWeightInit();

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
                new Layer(0, 4 ,ActivationType.None),
                new Layer(1, 4, ActivationType.Tanh),
                new Layer(2, 4, ActivationType.Tanh),
                new Layer(3, 8, ActivationType.ReLU),
                new Layer(4, 4, ActivationType.Tanh)
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
        public Func<float, int, ActivationType, float> WeightInit { get; set; }

        // Actually not NetParameters but rather TrainingParameters..

        public float LearningRate { get; set; }
        public float LearningRateChange { get; set; }
        public int EpochCount { get; set; }

        #endregion

        #region helpers

        // in factory?
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
