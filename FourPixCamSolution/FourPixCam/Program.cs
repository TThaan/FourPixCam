using System;
using System.Collections.Generic;
using static FourPixCam.Logger;

namespace FourPixCam
{
    class Program
    {
        #region fields

        static int
            samplesCount = 250,
            trainingsCount = 5,
            epochCount = 10;
        static float 
            learningRate = 0.1f,
            aggregatedAccuracies = 0,
            meanAccuracy = 0,
            sampleTolerance = 0.2f,
            distortionDeviation = .3f;        
        static List<(int TrainingsNr, float Accuracy)> accuracies = new List<(int, float)>();

        #endregion

        static void Main(string[] args)
        {
            #region

            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion

            for (int i = 0; i < trainingsCount; i++)
            {
                Log($"\n                                        T r a i n i n g {i}\n", Display.ToConsoleAndFile);

                NeuralNet net = NeuralNetFactory.GetNeuralNet("Implement jsonSource later!", false);
                Trainer trainer = new Trainer(net.Log());

                Sample[] trainingData = DataFactory.GetTrainingData(samplesCount, sampleTolerance, distortionDeviation);
                Sample[] testingData = DataFactory.GetTestingData(2);

                float accuracy = trainer.Train(trainingData, testingData, learningRate, epochCount);

                accuracies.Add((i, accuracy)); 
                aggregatedAccuracies += accuracy;
            }

            meanAccuracy = aggregatedAccuracies / trainingsCount;
            Log($"\n                                             Finished {trainingsCount} trainings with a mean accuracy of {meanAccuracy}!\n", Display.ToConsoleAndFile);
            Log($"\n{accuracies.ToVerticalCollectionString()}                                             \n", Display.ToConsoleAndFile);

            Console.ReadLine();
        }
    }
}
