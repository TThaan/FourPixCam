using MatrixHelper;
using NNet_InputProvider;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static FourPixCam.Logger;

namespace FourPixCam
{
    public class Trainer
    {
        #region ctor & fields

        float _learningRate, _learningRateChange, _epochCount, currentAccuracy;

        public Trainer(NeuralNet net, NetParameters netParameters)
        {
            #region

            IsLogOn = true;
            StandardDisplay = Display.ToFile;

            #endregion

            Net = net.Log();//.GetCopy()
            _learningRate = netParameters.LearningRate;
            _learningRateChange = netParameters.LearningRateChange;
            _epochCount = netParameters.EpochCount;
        }

        #endregion

        internal NeuralNet Net { get; }

        #region internal

        public async Task<float> Train(Sample[] trainingSamples, Sample[] testingSamples, int observerGap)
        {
            currentAccuracy = 0;

            for (int epoch = 0; epoch < _epochCount; epoch++)
            {
                currentAccuracy = await TrainEpoch(trainingSamples, testingSamples, epoch, observerGap);
                _learningRate *= _learningRateChange;
                // trainingSamples.Shuffle();
            }

            return currentAccuracy;
        }

        #endregion

        #region helpers
        
        async Task<float> TrainEpoch(Sample[] trainingSamples, Sample[] testingSamples, int currentEpoch, int observerGap)
        {
            LogTitle("T R A I N I N G", '*');
            Log(_learningRate, nameof(_learningRate) + ": ");
            Log(currentEpoch, nameof(currentEpoch) + ": ");
            Log(_epochCount, nameof(_epochCount) + ": ");

            int gap = observerGap;

            for (int sampleNr = 0; sampleNr < trainingSamples.Length; sampleNr++)
            {
                // debug
                Sample s = trainingSamples[sampleNr];

                LogTitle("F E E D   F O R W A R D", '*');
                Log($"epoch/sample: {currentEpoch}/{sampleNr}");

                trainingSamples[sampleNr].Input.Log($"\nOriginal Sample Input/Output\n");
                Log("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                Net.FeedForward(trainingSamples[sampleNr].Input);

                LogTitle("B A C K P R O P A P A G A T I O N", '*');
                Log($"epoch/sample: {currentEpoch}/{sampleNr}");
                s.ExpectedOutput.Log("\nTarget\n");
                float totalCost = Net.CostFunction(Net.Layers.Last().Processed.Output, trainingSamples[sampleNr].ExpectedOutput);
                totalCost.Log("         Total Cost = ");
                Net.PropagateBack(trainingSamples[sampleNr].ExpectedOutput, trainingSamples[sampleNr]);
                Net.AdjustWeightsAndBiases(_learningRate);

                if (gap == observerGap)
                {
                    OnSomethingHappend($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sampleNr})");
                    gap = 0;
                }
                gap++;

                //var continue = await OnStepFinishedAsync($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sample})");
                // Task isPauseOver = OnPausedAsync($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sampleNr})");
                // isPauseOver.Wait();

                Log($"                                    Current Accuracy : {currentAccuracy}    (eta = {_learningRate})", Display.ToFile);
            }

            // _learningRate *= .9f;

            LogTitle("T e s t", '*');
            Log($"epoch: {currentEpoch}");

            return await Test(testingSamples);
        }
        async Task<float> Test(Sample[] testingSamples)
        {
            return await Task.Run(() =>
            {
                int bad = 0, good = 0;

                foreach (var sample in testingSamples)
                {
                    Net.FeedForward(sample.Input);
                    sample.ActualOutput = Net.Layers.Last().Processed.Output;

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
            });            
        }

        #endregion

        #region Events

        public delegate void SomethingHappendEventHandler(string whatHappend);
        public event SomethingHappendEventHandler SomethingHappend;
        void OnSomethingHappend(string whatHappend)
        {
            SomethingHappend?.Invoke(whatHappend);
        }

        public delegate Task PausedEventHandler(string pauseInfo);
        public event PausedEventHandler Paused;
        async Task OnPausedAsync(string pauseInfo)
        {
            await Task.Run(() =>
            {
                Paused?.Invoke(pauseInfo);
            });
        }

        #endregion
    }
}
