using FourPixCam;
using MatrixHelper;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNet_InputProvider;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetDemoTest
{
    [TestClass]
    public class TrainerTest
    {
        #region Example Data

        static readonly NeuralNet Net = new NeuralNet(
            new Layer[]
            {
                new Layer(0, 2, ActivationType.Tanh),
                new Layer(0, 3, ActivationType.ReLU)
                {
                    Weights = new Matrix(new float[,]
                        {
                            {.5f, .5f },
                            {.5f, .5f },
                            {.5f, .5f }
                        })
                },
                new Layer(0, 2, ActivationType.Tanh)
                {
                    Weights = new Matrix(new float[,]
                        {
                            {.5f, .5f, .5f },
                            {.5f, .5f, .5f }
                        })
                },
            },
            CostType.SquaredMeanError);
        static readonly Sample[] TrainingSamples = new Sample[]
        {
            new Sample()
            {
                Input = new Matrix(new float[] { -1, 1 }),
                ExpectedOutput = new Matrix(new float[] { 0, 1 })  // All identic inputs (1/1, 0/0 or -1/-1) vs different inputs.
                // Label = "Monochrome";
            }
        };

        #endregion

        #region Expected 

        #endregion

        #region Expected 

        #endregion

        #region Testing Activations

        [TestMethod]
        public void Test_SetInput()
        {
            Matrix expect_L1 = new Matrix(new float[] { -1, 1 });
        }
        [TestMethod]
        public void Test_Trainer_FeedForward()
        {
            Sample.Tolerance = .2f;
            float delta = .001f;

            // Input Layer

            Matrix expect_L0in = new Matrix(new float[] { -1, 1 });
            Net.Layers[0].Processed.SetInput(TrainingSamples[0].Input);
            Matrix actual_L0in = Net.Layers[0].Processed.Input;

            Matrix diff0 = expect_L0in - actual_L0in;
            Assert.IsTrue(diff0.All(x => x <= Math.Abs(delta)));

            Matrix expect_L0out = new Matrix(new float[] { });
            Net.Layers[0].Processed.SetOutput();
            Matrix actual_L0out = Net.Layers[0].Processed.Output;

            // Hidden Layer

            Matrix expect_L1in = expect_L0out;

            Net.Layers[1].Processed.SetInput(null);
            Matrix actual_L1in = Net.Layers[1].Processed.Input;

            Matrix diff1 = expect_L1in - actual_L1in;
            Assert.IsTrue(diff1.All(x => x <= Math.Abs(delta)));

            Matrix expect_L1out = new Matrix(new float[] { });
            Net.Layers[1].Processed.SetOutput();
            Matrix actual_L1out = Net.Layers[1].Processed.Output;

            // Hidden Layer 

            Matrix expect_L2in = expect_L1out;

            Net.Layers[2].Processed.SetInput(null);
            Matrix actual_L2in = Net.Layers[2].Processed.Input;

            Matrix diff2 = expect_L2in - actual_L2in;
            Assert.IsTrue(diff2.All(x => x <= Math.Abs(delta)));

            Matrix expect_L2out = new Matrix(new float[] { });
            Net.Layers[2].Processed.SetOutput();
            Matrix actual_L2out = Net.Layers[2].Processed.Output;
        }

        #endregion
    }
}
