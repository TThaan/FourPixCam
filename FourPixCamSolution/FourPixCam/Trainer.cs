using System;
using System.Linq;

namespace FourPixCam
{
    /// <summary>
    /// 1st: Binary/Dichotom input
    /// 2nd: grey scales
    /// </summary>
    public class Trainer
    {
        #region ctor & fields

        Random rnd;
        float currentAccuracy;
        LearningNet learningNet;

        public Trainer(NeuralNet net)
        {
            Net = net;
            learningNet = new LearningNet(net);

            rnd = RandomProvider.GetThreadRandom();
        }

        #endregion

        #region properties

        public NeuralNet Net { get; }

        #endregion

        #region methods
        
        public void Train(Sample[] trainingData, Sample[] testingData, float learningRate, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                currentAccuracy = TrainEpoch(trainingData.ToArray(), testingData, learningRate);
                learningRate *= .9f;   // This help to avoids oscillation as our accuracy improves.
            }

            // var testAccuracy = ((Test(new FiringNetwork(Network), testingData) * 100).ToString("N1") + "%");
            // TrainingInfo += $"\r\nTotal epochs = {CurrentEpoch}\r\nFinal test accuracy = {testAccuracy}";

            Console.WriteLine("Finished training.");
        }

        float TrainEpoch(Sample[] trainingSet, Sample[] testingData, double learningRate)
        {
            Shuffle(trainingSet);

            for (int sample = 0; sample < trainingSet.Length; sample++)
            {
                var output = learningNet.FeedForwardAndGetOutput(trainingSet[sample].Input);
                // trainingSet[sample].IsOutputCorrect(output);
                learningNet.BackPropagate(trainingSet[sample].ExpectedOutput);    // I.e.: adjust weights and biases.
            }

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
            int bad = 0, good = 0;
            foreach (var sample in testingData)
            {
                var output = learningNet.FeedForwardAndGetOutput(sample.Input);
                if (sample.IsOutputCorrect(output))
                    good++;
                else
                    bad++;
            }
            return (float)good / (good + bad);
        }

        #endregion
    }
}
