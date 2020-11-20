using MatrixHelper;
using System;
using System.IO;
using System.Linq;

namespace FourPixCam
{
    public class Trainer
    {
        #region ctor & fields

        Random rnd;
        float currentAccuracy;
        LearningNet learningNet;

        public Trainer(NeuralNet net)
        {
            InitialNet = net.GetCopy();
            learningNet = new LearningNet(net);

            rnd = RandomProvider.GetThreadRandom();
        }

        #endregion

        #region properties

        public NeuralNet InitialNet { get; }

        #endregion

        #region methods
        
        public float Train(Sample[] trainingData, Sample[] testingData, float learningRate, int epochs, TextWriter standard, TextWriter custom)
        {
            // Test 

            Console.WriteLine($"                                    Initial Accuracy : {Test(testingData)}");

            var start = DateTime.Now;
            Console.WriteLine($"\n\n                                        Training Start: {start}");
            
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                $"T R A I N I N G".WriteDumpingTitle(
                    nameof(learningRate), learningRate,
                    nameof(epoch), epoch,
                    nameof(epochs), epochs);

                currentAccuracy = TrainEpoch(trainingData, testingData, learningRate, epoch);
                Console.WriteLine($"                                    Current Accuracy : {currentAccuracy}    (eta = {learningRate})");
                Console.SetOut(standard);
                Console.WriteLine($"                                    Current Accuracy : {currentAccuracy}    (eta = {learningRate})");
                Console.SetOut(custom);

                learningRate *= .9f;
            }

            Console.WriteLine("Finished training.");

            var end = DateTime.Now;
            Console.WriteLine($"\n\n                                        Training End: {end}  (Duration: {end - start})\n");

            // OriginalNet.DumpToConsole(true);
            // learningNet.Net.DumpToConsole(true);

            // Log/Dump
            for (int i = 1; i < InitialNet.W.Length; i++)
            {
                InitialNet.W[i].DumpToConsole($"\nStart  W{i}", true);
                Console.WriteLine($"Whole w-layer-deviation (from 0): {InitialNet.W[i].Sum() / InitialNet.W[i].LongCount()}");
                learningNet.Net.W[i].DumpToConsole($"\nEnd   W{i}", true);
                (learningNet.Net.W[i] - InitialNet.W[i]).DumpToConsole($"\nTotal dW{i}", true);

                // InitialNet.B[i].DumpToConsole($"\nStart  B{i}", true);
                // learningNet.Net.B[i].DumpToConsole($"\nEnd   B{i}", true);
                // (learningNet.Net.B[i] - InitialNet.B[i]).DumpToConsole($"\nTotal dB{i}", true);
            }

            return currentAccuracy;
        }

        float TrainEpoch(Sample[] trainingSet, Sample[] testingData, float learningRate, int epoch)
        {
            Shuffle(trainingSet);

            for (int sample = 0; sample < trainingSet.Length; sample++)
            {
                $"F E E D   F O R W A R D".WriteDumpingTitle($"epoch/sample: {epoch}/{sample}");
                trainingSet[sample].Input.DumpToConsole($"\nA[0] = ", true);
                Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                var output = learningNet.FeedForwardAndGetOutput(trainingSet[sample].Input);
                // trainingSet[sample].IsOutputCorrect(output);

                $"B A C K P R O P A P A G A T I O N".WriteDumpingTitle($"epoch/sample: {epoch}/{sample}");
                learningNet.BackPropagate(trainingSet[sample].ExpectedOutput.DumpToConsole("\nt =", true), learningRate);   
            }

            "T e s t".WriteDumpingTitle($"epoch: {epoch}");
            return Test(testingData);
        }
        /*
        double TrainEpoch(double learningRate)
        {
            Shuffle(trainingData);   // For each training epoch, randomize order of the training samples.

            foreach (double[] inputValues in trainingData)
            {
                double[] totalOutput = Net.GetTotalOutput(inputValues);
                //, expectedOutputOf

                // backpropagation (refactor! in Trainer?):
                foreach (Layer layer in Net.Reverse())  // .ToArray()
                {
                    double[] outputVotes = totalOutput;//7.Select()

                    for (int i = 0; i < layer.Count(); i++)
                    {
                        // if layer == output layer
                        if (layer.ID == Net.Count() - 1)
                        {
                            // For neurons in the output layer, the loss vs output slope = -error.
                            // layer.Neurons[i].OutputVotes = expectedOutputOf[neuron.Neuron.Index] - neuron.LastOutput;
                        }


                    }
                }
            }

            return Test(new FiringNet(Net), trainingData.Take(10000).ToArray()) * 100;
        }*/
        void Shuffle(Sample[] trainingData)
        {
            int n = trainingData.Length;

            while (n > 1)
            {
                int k = rnd.Next(n--);

                // Exchange arr[n] with arr[k]

                Sample temp = trainingData.ElementAt(n);
                trainingData[n] = trainingData[k];
                trainingData[k] = temp;
            }
        }
        float Test(Sample[] testingData)
        {
            Shuffle(testingData);

            int bad = 0, good = 0;

            foreach (var sample in testingData)
            {
                var output = learningNet.FeedForwardAndGetOutput(sample.Input, false);
                bool isCorrect;

                // param 1 = output (matrix)
                // param 2 = tolerance (float)
                if (sample.IsOutputCorrect(output, 0.2f))
                {
                    good++;
                    isCorrect = true; 
                }
                else
                { 
                    bad++;
                    isCorrect = false;
                }
                sample.DumpToConsole(output, isCorrect, true);
            }
            return (float)good / (good + bad);
        }

        #endregion
    }
}
