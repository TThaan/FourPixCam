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
            learningRate = 0.1f,
            sampleTolerance = 0.2f,
            distortionDeviation = .3f;

        public Initializer(NetParameters netParameters)
        {
            this.netParameters = netParameters ?? throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} ({GetType().Name}.ctor)");
        }
        public Initializer(NetParameters netParameters, string dataAsJson)
            :this(netParameters)
        {
            
        }
        public Initializer(NetParameters netParameters, List<IFormFile> files)
            : this(netParameters)
        {
            long size = files.Sum(x => x.Length);
            var filePaths = new List<string>();
            foreach (var formFile in files)
            {
                if (formFile.Length > 0)
                {
                    string filePath = Path.Combine(Directory.GetCurrentDirectory(), "UploadedFiles", formFile.FileName);
                    filePaths.Add(filePath);

                    using (var stream = new FileStream(filePath, FileMode.Create))
                    {
                        // await formFile.CopyToAsync(stream);
                        formFile.CopyTo(stream);
                    }
                }
            }
        }

        #endregion

        public void Run(bool turnBased = false)
        {
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
            Sample[] trainingData = DataFactory.GetTrainingData(samplesCount, sampleTolerance, distortionDeviation);
            Sample[] testingData = DataFactory.GetTestingData(2);

            trainer.Train(trainingData, testingData, learningRate, epochCount);
        }
    }
}
