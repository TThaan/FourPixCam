using FourPixCam.Factories;
using NNet_InputProvider;
using System;

namespace FourPixCam
{
    /// <summary>
    /// Entry for client.
    /// </summary>
    public class Initializer
    {
        #region ctor & fields

        NetParameters _netParameters;
        SampleSetParameters _sampleSetParameters;

        public Initializer()//NetParameters netParameters, SampleSetParameters sampleSetParameters
        {
            #region Parameter Checks

            //_netParameters = netParameters ?? throw new NullReferenceException(
            //        $"{typeof(NetParameters).Name} {nameof(netParameters)} ({typeof(Initializer).Name}.ctor)");
            //_sampleSetParameters = sampleSetParameters ?? throw new NullReferenceException(
            //           $"{typeof(SampleSetParameters).Name} {nameof(sampleSetParameters)} ({typeof(Initializer).Name}.ctor)");

            #endregion

            // Net = NeuralNetFactory.GetNeuralNet(_netParameters);
            // Samples = Creator.GetSampleSet(_sampleSetParameters);
            // Trainer = new Trainer(Net, _netParameters);
        }
        //public Initializer()//NetParameters netParameters, SampleSet samples
        //{
        //    #region Parameter Checks

            // _netParameters = netParameters ?? throw new NullReferenceException(
            //         $"{typeof(NetParameters).Name} {nameof(netParameters)} ({typeof(Initializer).Name}.ctor)");
            // Samples = samples ?? throw new NullReferenceException(
            //            $"{typeof(SampleSet).Name} {nameof(samples)} ({typeof(Initializer).Name}.ctor)");

            #endregion
        //
        //    // Net = NeuralNetFactory.GetNeuralNet(_netParameters);
        //    // Trainer = new Trainer(Net, _netParameters);
        //}


        // #endregion

        #region public

        public NeuralNet Net { get; set; }  // as lib?
        public SampleSet Samples { get; set; }
        public Trainer Trainer { get; set; }
        public NeuralNet GetNeuralNet(NetParameters netParameters)
        {
            return NeuralNetFactory.GetNeuralNet(netParameters); ;
        }

        #endregion
    }
}
