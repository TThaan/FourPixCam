using System;
using System.Collections.Generic;
using System.IO;

namespace FourPixCam
{
    class Program
    {
        #region fields

        static int
            samplesCount = 500,
            trainingsCount = 10,
            epochCount = 20;
        static float 
            learningRate = 0.1f,
            aggregatedAccuracies,
            meanAccuracy = 0;        
        static List<(int TrainingsNr, float Accuracy)> accuracies = new List<(int, float)>();
        static StreamWriter customWriter = GetCustomWriter(@"c:\temp\FourPixTest.txt");
        static TextWriter standardWriter = Console.Out;

        #endregion

        static void Main(string[] args)
        {
            aggregatedAccuracies = 0;

            for (int i = 0; i < trainingsCount; i++)
            {
                Console.SetOut(standardWriter);
                Console.WriteLine($"\n                                        T r a i n i n g {i}\n");
                Console.SetOut(customWriter);
                Console.WriteLine($"\n\n\n                                        T r a i n i n g {i}\n\n");
                // Console.SetError(customWriter);

                NeuralNet net = NeuralNetFactory.GetNeuralNet("Implement jsonSource later!", false);
                Trainer trainer = new Trainer(net);
                //net.DumpToExplorer();
                net.DumpToConsole(true);

                Sample[] trainingData = DataFactory.GetTrainingData(samplesCount);
                Sample[] testingData = DataFactory.GetTestingData(2);

                float accuracy = trainer.Train(trainingData, testingData, learningRate, epochCount, standardWriter, customWriter);

                accuracies.Add((i, accuracy)); 
                aggregatedAccuracies += accuracy;
            }

            meanAccuracy = aggregatedAccuracies / trainingsCount;
            Console.SetOut(customWriter);
            Console.WriteLine($"\n                                             Finished {trainingsCount} trainings with a mean accuracy of {meanAccuracy}!\n");
            Console.SetOut(standardWriter);
            Console.WriteLine($"\n                                             Finished {trainingsCount} trainings with a mean accuracy of {meanAccuracy}!\n");

            Console.SetOut(customWriter);
            Console.WriteLine(accuracies.ToVerticalCollectionString());
            Console.WriteLine();
            Console.SetOut(standardWriter);
            Console.WriteLine(accuracies.ToVerticalCollectionString());
            Console.WriteLine();

            Console.ReadLine();
        }

        #region helpers

        static StreamWriter GetCustomWriter(string path)
        {
            FileStream filestream = new FileStream(path, FileMode.Create);
            StreamWriter customWriter = new StreamWriter(filestream);
            customWriter.AutoFlush = true;

            return customWriter;
        }

        #endregion
    }
}
