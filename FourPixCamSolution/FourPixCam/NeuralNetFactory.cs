using System;

namespace FourPixCam
{
    internal class NeuralNetFactory
    {
        #region fields

        readonly Random rnd = RandomProvider.GetThreadRandom();

        #endregion

        #region ctor
        #endregion

        #region properties
        #endregion

        #region methods

        public static NeuralNet GetNeuralNet(string jsonSource)
        {
            return new NeuralNet(4,4,4,8,4);
        }

        #region helper methods

        

        #endregion

        #endregion
    }
}
