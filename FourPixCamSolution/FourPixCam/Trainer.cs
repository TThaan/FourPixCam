using MatrixHelper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace FourPixCam
{
    /// <summary>
    /// 1st: Binary/Dichotom input
    /// 2nd: grey scales
    /// </summary>
    public class Trainer
    {
        #region fields

        Random rnd;
        public float[][] trainingData;
        public Dictionary<float[], float[]> expectedOutputOf;
        public Dictionary<float[], string> expectedResultOf;
        // float currentAccuracy;

        LearningNet learningNet;

        #endregion

        #region ctor

        public Trainer(NeuralNet net)
        {
            Net = net;
            learningNet = new LearningNet(net);

            rnd = RandomProvider.GetThreadRandom();
            trainingData = GetTrainingData(100);
            expectedOutputOf = GetExpectedOutput();
            expectedResultOf = GetExpectedResult();
        }

        #region helper methods

        float[][] GetTrainingData(int sampleSize)
        {
            return Enumerable.Range(0, sampleSize)
                .Select(x => new float[]
                {
                    (float)Math.Round(rnd.NextDouble()),
                    (float)Math.Round(rnd.NextDouble()),
                    (float)Math.Round(rnd.NextDouble()),
                    (float)Math.Round(rnd.NextDouble())
                })
                .ToArray();
        }
        Dictionary<float[], float[]> GetExpectedOutput()
        {
            Dictionary<float[], float[]> result = new Dictionary<float[], float[]>();

            foreach (float[] trainingSample in trainingData)
            {
                // result = dictionary where key = value,
                // but that need'nt be the case in other/more general neural nets.
                result[trainingSample] = trainingSample;
            }

            return result;
        }
        Dictionary<float[], string> GetExpectedResult()
        {
            Dictionary<float[], string> result = new Dictionary<float[], string>();

            foreach (float[] trainingSample in trainingData)
            {
                result[trainingSample] = GetLabel(trainingSample);
            }

            return result;
        }
        string GetLabel(float[] sample)
        {
            float _0 = sample[0];
            float _1 = sample[1];
            float _2 = sample[2];
            float _3 = sample[3];

            if (_0 == _1)
            {
                if (_2 == _3)
                {
                    if (_0 == _2)
                    {
                        return _0 == 0
                            ? "AllBlack"
                            : "AllWhite";
                    }
                    else
                    {
                        return _0 == 0
                            ? "Black Top - White Bottom (hori)"
                            : "White Top - Black Bottom (hori)";
                    }
                }
            }
            else if (_0 == _2)
            {
                if (_1 == _3)
                {
                    return _0 == 0
                        ? "Black Left - White Right (vert)"
                        : "White Left - Black Right (vert)";
                }
            }
            else if (_0 == _3)
            {
                if (_1 == _2)
                {
                    return _0 == 0
                        ? "Black TopLeft & RightBottom (diag)"
                        : "White TopLeft & RightBottom (diag)";
                }
            }

            return "No match.";
        }

        #endregion

        #endregion

        #region properties

        public NeuralNet Net { get; }

        #endregion

        #region methods
        
        public void Train(float learningRate, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int sample = 0; sample < trainingData.Length; sample++)
                {
                    learningNet.FeedForward(new Matrix(trainingData[sample]));
                    Matrix t = new Matrix(expectedOutputOf[trainingData[sample]]);
                    // Matrix cost = learningNet.GetTotalCostOfLastSample(t);
                    learningNet.BackPropagate(t);    // I.e.: adjust weights and biases.
                }

                //currentAccuracy = TrainEpoch(learningRate);
                learningRate *= .9f;   // This help to avoids oscillation as our accuracy improves.
            }
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
        void Shuffle(float[][] trainingData)
        {
            int n = trainingData.Length;
            while (n > 1)
            {
                int k = rnd.Next(n--);
                float[] temp = trainingData[n];
                trainingData[n] = trainingData[k];
                trainingData[k] = temp;
            }
        }

        #endregion
    }
}
