using System;
using System.Linq;

namespace FourPixCam
{
    public class ProcessingNet : NeuralNet
    {
        #region ctor & fields

        NeuralNet _net;
        Layer[] _layers;

        public ProcessingNet(NeuralNet net)
            :base(null)
        {
            _net = net ??
                throw new NullReferenceException($"{typeof(NeuralNet).Name} {nameof(net)} " +
                $"({GetType().Name}.ctor)");
            _layers = net.Layers.ToArray();
        }

        #endregion

        #region public

        #endregion

        #region helpers

        #endregion
    }
}
