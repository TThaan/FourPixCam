using MatrixHelper;
using System;
using System.Linq;
using static FourPixCam.Logger;

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
        
        public float Train(Sample[] trainingData, Sample[] testingData, float learningRate, int epochs)
        {
            var start = DateTime.Now;
            Log($"\n\n                                        Training Start: {start}");

            // Initial Test 
            Log($"                                    Initial Accuracy : {Test(testingData)}");
            
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                LogTitle("T R A I N I N G", '*');
                Log(learningRate, nameof(learningRate));
                Log(epoch, nameof(epoch));
                Log(epochs, nameof(epochs));

                currentAccuracy = TrainEpoch(trainingData, testingData, learningRate, epoch);
                
                Log($"                                    Current Accuracy : {currentAccuracy}    (eta = {learningRate})", Display.ToConsoleAndFile);

                learningRate *= .9f;
            }

            Log("Finished training.");

            var end = DateTime.Now;
            Log($"\n\n                                        Training End: {end}  (Duration: {end - start})\n");

            // OriginalNet.Log(true);
            // learningNet.Net.Log(true);

            // Log/Dump
            for (int i = 1; i < InitialNet.W.Length; i++)
            {
                InitialNet.W[i].Log($"\nStart  W{i}");
                Log($"          Whole w-layer-deviation (from 0): {InitialNet.W[i].Sum() / InitialNet.W[i].LongCount()}");
                learningNet.Net.W[i].Log($"\nEnd   W{i}");
                (learningNet.Net.W[i] - InitialNet.W[i]).Log($"\nTotal dW{i}");

                // InitialNet.B[i].Log($"\nStart  B{i}", true);
                // learningNet.Net.B[i].Log($"\nEnd   B{i}", true);
                // (learningNet.Net.B[i] - InitialNet.B[i]).Log($"\nTotal dB{i}", true);
            }

            return currentAccuracy;
        }

        float TrainEpoch(Sample[] trainingSet, Sample[] testingData, float learningRate, int epoch)
        {
            for (int sample = 0; sample < trainingSet.Length; sample++)
            {
                LogTitle("F E E D   F O R W A R D", '*');
                Log($"epoch/sample: {epoch}/{sample}");

                trainingSet[sample].Input.Log($"\nA[0] = ");
                Log("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                var output = learningNet.FeedForwardAndGetOutput(trainingSet[sample].Input);

                LogTitle("B A C K P R O P A P A G A T I O N", '*');
                Log($"epoch/sample: {epoch}/{sample}");
                learningNet.BackPropagate(trainingSet[sample].ExpectedOutput.Log("\nt ="), learningRate);   
            }

            LogTitle("T e s t", '*');
            Log($"epoch: {epoch}");
            return Test(testingData);
        }
        
        float Test(Sample[] testingData)
        {
            int bad = 0, good = 0;

            foreach (var sample in testingData.Shuffle())
            {
                sample.ActualOutput = learningNet.FeedForwardAndGetOutput(sample.Input);
                
                if (sample.IsOutputCorrect == true)
                {
                    good++;
                }
                else
                { 
                    bad++;
                }
                sample.Log();
            }
            return (float)good / (good + bad);
        }

        #endregion
    }
}
