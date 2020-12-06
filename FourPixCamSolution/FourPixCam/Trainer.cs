using NNet_InputProvider;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace FourPixCam
{
    public class Trainer
    {
        #region ctor & fields

        float _learningRate, _learningRateChange, _epochCount, currentAccuracy;
        DateTime t1 = default, t2 = default, t3 = default, t4 = default, t5 = default;
        TimeSpan t12sum, t23sum, t34sum, t45sum, t51sum;

        internal Trainer(NeuralNet net, NetParameters netParameters)
        {
            Net = net;//.GetCopy()
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
            }

            return currentAccuracy;
        }

        #endregion

        #region helpers
        
        async Task<float> TrainEpoch(Sample[] trainingSamples, Sample[] testingSamples, int currentEpoch, int observerGap)
        {
            // observerGap = 100;
            int gap = observerGap;

            for (int sample = 0; sample < trainingSamples.Length; sample++)
            {
                #region debugging

                t1 = DateTime.Now;
                t51sum += t1 - t5;

                Net.FeedForward(trainingSamples[sample].Input);
                t2 = DateTime.Now;
                Net.PropagateBack(trainingSamples[sample].ExpectedOutput, _learningRate);
                t3 = DateTime.Now;

                #endregion

                if (gap == observerGap)
                {
                    OnSomethingHappend($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sample})");
                    gap = 0;
                }
                gap++;

                // debugging
                t4 = DateTime.Now;

                //var continue = await OnStepFinishedAsync($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sample})");
                Task isPauseOver = OnPausedAsync($"Accuracy: {currentAccuracy} (Epoch: {currentEpoch}, Sample: {sample})");
                isPauseOver.Wait();

                // debugging
                t5 = DateTime.Now;

                #region debugging

                t12sum += t2 - t1;
                t23sum += t3 - t2;
                t34sum += t4 - t3;
                t45sum += t5 - t4;

                if (sample > 100)
                {
                    // ..
                }

                #endregion
            }

            return await Test(testingSamples);
        }
        async Task<float> Test(Sample[] testingSamples)
        {
            return await Task.Run(() =>
            {
                int bad = 0, good = 0;

                foreach (var sample in testingSamples.Shuffle())
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
