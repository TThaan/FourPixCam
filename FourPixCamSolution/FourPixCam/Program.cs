using System;

namespace FourPixCam
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet net = NeuralNetFactory.GetNeuralNet("Implement jsonSource later!");
            Trainer trainer = new Trainer(net);
            //net.DumpToExplorer();
            net.DumpToConsole();

            Sample[] trainingData = DataFactory.GetTrainingData(100);
            Sample[] testingData = DataFactory.GetTestingData();
            trainer.Train(trainingData, testingData, 0.02f, 10);

            Console.ReadLine();

            // net.DumpToConsole(true);
            // var test = net.GetTotalOutput(trainer.trainingData.First());
        }
    }
}
