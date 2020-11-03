using System;

namespace FourPixCam
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet net = NeuralNetFactory.GetNeuralNet("");
            Trainer trainer = new Trainer(net);

            //string debug;
            //net.DumpToHTMLDebugger(out debug);
            //net.DumpToConsole(true);

            net.DumpToExplorer();
            trainer.Train(0.02f, 10);

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
