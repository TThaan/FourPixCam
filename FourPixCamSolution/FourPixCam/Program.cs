using System;

namespace FourPixCam
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet net = NeuralNetFactory.GetNeuralNet("");
            Trainer trainer = new Trainer(net);
            net.DumpToExplorer();

            var trainigData = DataFactory.GetTrainingData(100);
            var testingData = DataFactory.GetTrainingData(10);
            trainer.Train(trainigData, testingData, 0.02f, 10);

            Console.ReadLine();

            // net.DumpToConsole(true);
            // var test = net.GetTotalOutput(trainer.trainingData.First());

            /* debug

            Matrix A = new Matrix(
                new[,] {
                    { (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble() },
                    { (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble() },
                    { (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble() },
                    { (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble() },
                    { (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble(), (float)RandomProvider.GetThreadRandom().NextDouble() }
                });
            Matrix B = new Matrix(
                new[,] {
                    { (float)RandomProvider.GetThreadRandom().NextDouble()},
                    { (float)RandomProvider.GetThreadRandom().NextDouble()},
                    { (float)RandomProvider.GetThreadRandom().NextDouble()}
                });
            // var v = Matrix.ElementWiseProduct(A, B);

            */
        }
    }
}
