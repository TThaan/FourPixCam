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
        #region ctor & fields

        static NetParameters _netParameters;
        static SampleSetParameters _sampleSetParameters;
        static DateTime start, end;
        static TimeSpan duration;
        // static int epochCount = 10;

        #endregion

        public static void Run(NetParameters netParameters, SampleSetParameters sampleSetParameters)
        {
            // define data
            //_trainingData = GetData(trainingData);  // in DataFactory -> possible: 1d&2d(&3d) byte arrays, img, 
            //_testingData = GetData(testingData);    // in DataFactory

            NeuralNet net = NeuralNetFactory.GetNeuralNet(netParameters);

            PrepareTraining(netParameters, sampleSetParameters);
            Train(net);

        }
        public static void RunTurnBased(NetParameters netParameters, SampleSetParameters sampleSetParameters)
        {
            throw new NotImplementedException();
        }

        static void Train(NeuralNet net)   // data(Parameters) as parameter?
        {
            start = DateTime.Now.Log("\n\n                                        Training Start: ", Display.ToConsoleAndFile);
            Log($"\n                                        T r a i n i n g \n", Display.ToConsoleAndFile);

            Trainer trainer = new Trainer(net.Log());

            // FacPat:
            var _trainingData = new NNet_InputProvider.FourPixCam.DataFactory().TrainingSamples;
            var _testingData = new NNet_InputProvider.FourPixCam.DataFactory().TestingSamples;

            trainer.Train(_trainingData, _testingData, _netParameters.LearningRate, _netParameters.EpochCount);

            end = DateTime.Now.Log($"\n\n                                        Training End: \n", Display.ToConsoleAndFile);
            duration = (end - start).Log($"Duration: ", Display.ToConsoleAndFile);
        }

        #region helpers

        static void PrepareTraining(NetParameters netParameters, SampleSetParameters sampleSetParameters)
        {
            _netParameters = netParameters ?? throw new NullReferenceException(
                       $"{typeof(NetParameters).Name} {nameof(netParameters)} ({typeof(Initializer).Name}.{nameof(PrepareTraining)})");
            _sampleSetParameters = sampleSetParameters ?? throw new NullReferenceException(
                       $"{typeof(SampleSetParameters).Name} {nameof(sampleSetParameters)} ({typeof(Initializer).Name}.{nameof(PrepareTraining)})");

            #region Logger

            ConsoleAllocator.ShowConsoleWindow();
            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion


        }

        #endregion
    }
}
