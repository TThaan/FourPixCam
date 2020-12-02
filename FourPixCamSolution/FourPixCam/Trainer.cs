using NNet_InputProvider;
using static FourPixCam.Logger;

namespace FourPixCam
{
    internal class Trainer
    {
        #region ctor & fields

        float _learningRate, _learningRateChange, _epochCount;
        NeuralNet InitialNet { get; }
        //ProcessingNet processingNet;

        internal Trainer(NeuralNet net, float learningRate, float learningRateChange, int epochCount)
        {
            InitialNet = net;//.GetCopy()
            //processingNet = new ProcessingNet();//net
            _learningRate = learningRate;
            _learningRateChange = learningRateChange;
            _epochCount = epochCount;
        }

        #endregion

        #region internal

        internal float Train(Sample[] trainingSamples, Sample[] testingSamples)
        {
            float currentAccuracy = 0;

            // Initial Test 
            Log($"                                    Initial Accuracy : {Test(testingSamples)}");
            
            for (int epoch = 0; epoch < _epochCount; epoch++)
            {
                LogTitle("T R A I N I N G", '*');
                Log(_learningRate, nameof(_learningRate));
                Log(epoch, nameof(epoch));
                Log(_epochCount, nameof(_epochCount));

                currentAccuracy = TrainEpoch(trainingSamples, testingSamples, epoch);                
                Log($"                                    Current Accuracy : {currentAccuracy}    (eta = {_learningRate})", Display.ToConsoleAndFile);
                _learningRate *= _learningRateChange;
            }

            Log("Finished training.", Display.ToConsoleAndFile);
            return currentAccuracy;
        }

        #endregion

        #region helpers

        float TrainEpoch(Sample[] trainingSamples, Sample[] testingSamples, int currentEpoch)
        {
            for (int sample = 0; sample < trainingSamples.Length; sample++)
            {
                LogTitle("F E E D   F O R W A R D", '*');
                Log($"epoch/sample: {currentEpoch}/{sample}");

                trainingSamples[sample].Input.Log($"\nA[0] = ");
                Log("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                var output = InitialNet.FeedForward(trainingSamples[sample].Input);

                LogTitle("B A C K P R O P A P A G A T I O N", '*');
                Log($"epoch/sample: {currentEpoch}/{sample}");
                InitialNet.PropagateBack(trainingSamples[sample].ExpectedOutput.Log("\nt ="), _learningRate);
                // InitialNet.AdaptWeightsAndBiases(learningRate);
            }

            LogTitle("T e s t", '*');
            Log($"epoch: {_epochCount}");
            return Test(testingSamples);
        }
        float Test(Sample[] testingSamples)
        {
            int bad = 0, good = 0;

            foreach (var sample in testingSamples.Shuffle())
            {
                sample.ActualOutput = InitialNet.FeedForward(sample.Input);

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
