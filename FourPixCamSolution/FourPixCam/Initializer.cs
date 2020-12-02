using NNet_InputProvider;
using System;
using static FourPixCam.Logger;

namespace FourPixCam
{
    /// <summary>
    /// Entry for client.
    /// </summary>
    public class Initializer
    {
        public static void Run(NetParameters netParameters, SampleSetParameters sampleSetParameters)
        {
            #region Parameter Checks

            if (netParameters == null) 
                throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} ({typeof(Initializer).Name}.{nameof(Run)})");
            if(sampleSetParameters == null)
                throw new NullReferenceException(
                       $"{typeof(SampleSetParameters).Name} {nameof(sampleSetParameters)} ({typeof(Initializer).Name}.{nameof(Run)})");

            #endregion

            #region Logger

            ConsoleAllocator.ShowConsoleWindow();
            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion

            NeuralNet net = NeuralNetFactory.GetNeuralNet(netParameters);
            SampleSet sampleSet = Creator.GetSampleSet(sampleSetParameters);
            Trainer trainer = new Trainer(net.Log(), 
                netParameters.LearningRate, netParameters.LearningRateChange, netParameters.EpochCount);

            var start = DateTime.Now.Log("\n\n                                        Training Start: ", Display.ToConsoleAndFile);
            Log($"\n                                        T r a i n i n g \n", Display.ToConsoleAndFile);
            
            trainer.Train(sampleSet.TrainingSamples, sampleSet.TestingSamples);

            var end = DateTime.Now.Log($"\n\n                                        Training End: \n", Display.ToConsoleAndFile);
            var duration = (end - start).Log($"Duration: ", Display.ToConsoleAndFile);
        }
        public static void RunTurnBased(NetParameters netParameters, SampleSetParameters sampleSetParameters)
        {
            throw new NotImplementedException();
        }
    }
}
