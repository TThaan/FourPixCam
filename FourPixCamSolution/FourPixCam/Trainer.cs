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
                InitialNet.W[i].DumpToConsole($"\nStart  W{i}");
                Console.WriteLine($"Whole w-layer-deviation (from 0): {InitialNet.W[i].Sum() / InitialNet.W[i].LongCount()}");
                learningNet.Net.W[i].DumpToConsole($"\nEnd   W{i}");
                (learningNet.Net.W[i] - InitialNet.W[i]).DumpToConsole($"\nTotal dW{i}");

                // InitialNet.B[i].DumpToConsole($"\nStart  B{i}", true);
                // learningNet.Net.B[i].DumpToConsole($"\nEnd   B{i}", true);
                // (learningNet.Net.B[i] - InitialNet.B[i]).DumpToConsole($"\nTotal dB{i}", true);
            }

            return currentAccuracy;
        }

        float TrainEpoch(Sample[] trainingSet, Sample[] testingData, float learningRate, int epoch)
        {
            for (int sample = 0; sample < trainingSet.Length; sample++)
            {
                $"F E E D   F O R W A R D".WriteDumpingTitle($"epoch/sample: {epoch}/{sample}");
                trainingSet[sample].Input.DumpToConsole($"\nA[0] = ");
                Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                var output = learningNet.FeedForwardAndGetOutput(trainingSet[sample].Input);
                // trainingSet[sample].IsOutputCorrect(output);

                $"B A C K P R O P A P A G A T I O N".WriteDumpingTitle($"epoch/sample: {epoch}/{sample}");
                learningNet.BackPropagate(trainingSet[sample].ExpectedOutput.DumpToConsole("\nt ="), learningRate);   
            }

            "T e s t".WriteDumpingTitle($"epoch: {epoch}");
            return Test(testingData);
        }
        
        float Test(Sample[] testingData)
        {
            int bad = 0, good = 0;

            foreach (var sample in testingData.Shuffle())
            {
                // sample.HasBeenCheckedAlready = false;
                sample.ActualOutput = learningNet.FeedForwardAndGetOutput(sample.Input);
                
                if (sample.IsOutputCorrect == true)
                {
                    good++;
                }
                else
                { 
                    bad++;
                }
                sample.LogIt();
            }
            return (float)good / (good + bad);
        }

        #endregion

        //bool IsTrue(bool? check)
        //{
        //    return check.HasValue && check.Value;
        //}
    }
}
