using System;
using System.IO;

namespace FourPixCam
{
    class Program
    {
        static void Main(string[] args)
        {
            FileStream filestream = new FileStream(@"c:\temp\FourPixTest.txt", FileMode.Create);
            var streamwriter = new StreamWriter(filestream);
            streamwriter.AutoFlush = true;
            Console.SetOut(streamwriter);
            Console.SetError(streamwriter);

            NeuralNet net = NeuralNetFactory.GetNeuralNet("Implement jsonSource later!");
            Trainer trainer = new Trainer(net);
            //net.DumpToExplorer();
            net.DumpToConsole(true);

            Sample[] trainingData = DataFactory.GetTrainingData(1000);
            Sample[] testingData = DataFactory.GetTestingData(2);
            trainer.Train(trainingData, testingData, 0.02f, 10);

            // Console.ReadLine();

            // net.DumpToConsole(true);
            // var test = net.GetTotalOutput(trainer.trainingData.First());
        }
    }
}
