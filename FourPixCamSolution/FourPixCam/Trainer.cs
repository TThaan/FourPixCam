using NNet_InputProvider;
using static FourPixCam.Logger;

namespace FourPixCam
{
    public class Trainer
    {
        #region ctor & fields

        float currentAccuracy;
        NeuralNet InitialNet { get; }
        //ProcessingNet processingNet;

        public Trainer(NeuralNet net)
        {
            InitialNet = net;//.GetCopy()
            //processingNet = new ProcessingNet();//net
        }

        #endregion

        #region public
        
        public float Train(Sample[] trainingData, Sample[] testingData, float learningRate, int epochs)
        {
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

            Log("Finished training.", Display.ToConsoleAndFile);
            return currentAccuracy;
        }

        #endregion

        #region helpers

        float TrainEpoch(Sample[] trainingSet, Sample[] testingData, float learningRate, int epoch)
        {
            for (int sample = 0; sample < trainingSet.Length; sample++)
            {
                LogTitle("F E E D   F O R W A R D", '*');
                Log($"epoch/sample: {epoch}/{sample}");

                trainingSet[sample].Input.Log($"\nA[0] = ");
                Log("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");

                var output = InitialNet.FeedForward(trainingSet[sample].Input);

                LogTitle("B A C K P R O P A P A G A T I O N", '*');
                Log($"epoch/sample: {epoch}/{sample}");
                InitialNet.PropagateBack(trainingSet[sample].ExpectedOutput.Log("\nt ="), learningRate);
                // InitialNet.AdaptWeightsAndBiases(learningRate);
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
