using System;
using System.Collections.Generic;
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
            trainingsCount = 10,
            epochCount = 10;
        float
            learningRate = 0.1f,
            sampleTolerance = 0.2f,
            distortionDeviation = .3f;
        List<(int TrainingsNr, float Accuracy)> accuracies = new List<(int, float)>();

        public Initializer(NetParameters netParameters)
        {
            this.netParameters = netParameters ?? throw new NullReferenceException(
                    $"{typeof(NetParameters).Name} {nameof(netParameters)} ({GetType().Name}.ctor)");
        }

        #endregion

        public void Run()
        {
            #region Logger

            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion

            for (int i = 0; i < 10; i++)
            {
                learningRate *= 1.25f;
                TrainNTimes(trainingsCount);
            }

            Log($"\n{accuracies.ToVerticalCollectionString()}                                             \n", Display.ToConsoleAndFile);
            Console.ReadLine();
        }

        void TrainNTimes(int _trainingsCount)
        {
            float
                aggregatedAccuracies = 0,
                meanAccuracy = 0;

            for (int i = 0; i < _trainingsCount; i++)
            {
                Log($"\n                                        T r a i n i n g {i}\n", Display.ToConsoleAndFile);

                NeuralNet net = NeuralNetFactory.GetNeuralNet(netParameters);
                Trainer trainer = new Trainer(net.Log());

                Sample[] trainingData = DataFactory.GetTrainingData(samplesCount, sampleTolerance, distortionDeviation);
                Sample[] testingData = DataFactory.GetTestingData(2);

                float accuracy = trainer.Train(trainingData, testingData, learningRate, epochCount);

                accuracies.Add((i, accuracy));
                aggregatedAccuracies += accuracy;
            }

            meanAccuracy = aggregatedAccuracies / trainingsCount;
            Log($"\n                                             Finished {trainingsCount} trainings with a mean accuracy of {meanAccuracy}!\n", Display.ToConsoleAndFile);
        }
    }
}
