using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static FourPixCam.Logger;

namespace FourPixCam
{
    /// <summary>
    /// Entry for client.
    /// </summary>
    public class Initializer
    {
        #region ctor & fields

        NetParameters netParameters;
        int
            samplesCount = 250,
            epochCount = 10;
        float
            learningRate = .1f,
            sampleTolerance = .2f,
            distortionDeviation = .3f;
        Sample[] _trainingData, _testingData;
        
        public Initializer(NetParameters netParameters)
        {
            this.netParameters = netParameters ?? throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} ({GetType().Name}.ctor)");
        }
        public Initializer(NetParameters netParameters, string dataAsJson)
            :this(netParameters)
        {
            
        }

        #endregion

        /// <summary>
        /// Order:
        /// Url_TrainingLabels, string Url_TrainingImages, string Url_TestingLabels, string Url_TestingImages
        /// </summary>
        public void Run(bool turnBased = false, params string[] urls)
        {
            // define data
            //_trainingData = GetData(trainingData);  // in DataFactory -> possible: 1d&2d(&3d) byte arrays, img, 
            //_testingData = GetData(testingData);    // in DataFactory



            #region Logger

            ConsoleAllocator.ShowConsoleWindow();
            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion

            var start = DateTime.Now.Log("\n\n                                        Training Start: ", Display.ToConsoleAndFile);

            NeuralNet net = NeuralNetFactory.GetNeuralNet(netParameters);
            Train(net);

            var end = DateTime.Now.Log($"\n\n                                        Training End: \n", Display.ToConsoleAndFile);
            var duration = (end - start).Log($"Duration: ", Display.ToConsoleAndFile);
        }

        void Train(NeuralNet net)   // data(Parameters) as parameter?
        {
            Log($"\n                                        T r a i n i n g \n", Display.ToConsoleAndFile);

            Trainer trainer = new Trainer(net.Log());
            // _trainingData = DataFactory.GetTrainingData(samplesCount, sampleTolerance, distortionDeviation);
            // _testingData = DataFactory.GetTestingData(2);

            trainer.Train(_trainingData, _testingData, learningRate, epochCount);
        }
    }
}
