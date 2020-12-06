using FourPixCam.Activators;
using MatrixHelper;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace NeuralNetDemoTest
{
    [TestClass]
    public class ActivatorTest
    {
        #region Example Data

        static readonly Matrix WeightedInput = new Matrix
            (
                new float[] { .8f, 0, -.6f, 1 }
            );
        static readonly Matrix WeightedInput_BackUp = new Matrix
            (
                new float[] { .8f, 0, -.6f, 1 }
            );

        #endregion

        #region Expected Activations

        static readonly Matrix Expected_NullActivator_Activation = new Matrix
            (
                new float[] { .8f, 0, -.6f, 1 }
            );
        static readonly Matrix Expected_ReLU_Activation = new Matrix
            (
                new float[] { .8f, 0, 0, 1 }
            );
        static readonly Matrix Expected_LeakyReLU_Activation = new Matrix
            (
                new float[] { .8f, 0, -.006f, 1 }
            );
        static readonly Matrix Expected_Sigmoid_Activation = new Matrix
            (
                new float[] { .6899744811f, .5f, .3543436938f, .7310585786f }//.6899744811, 
            );
        static readonly Matrix Expected_Tanh_Activation = new Matrix
            (
                new float[] { .6640367703f, 0, -.537049567f, .761594156f }
            );
        static readonly Matrix Expected_SoftMax_Activation = new Matrix
            (
                new float[] { .6666666667f, 0, -.5f, .8333333333f }
            );
        static readonly Matrix Expected_SoftMaxwithCEL_Activation = new Matrix
            (
                new float[] { .6666666667f, 0, -.5f, .8333333333f }
            );

        #endregion

        #region Expected Derivations

        static readonly Matrix Expected_NullActivator_Derivation = new Matrix
            (
                new float[] { 1, 1, 1, 1 }
            );
        static readonly Matrix Expected_ReLU_Derivation = new Matrix
            (
                new float[] { 1, 1, 0, 1 }
            );
        static readonly Matrix Expected_LeakyReLU_Derivation = new Matrix
            (
                new float[] { 1, 1, .001f, 1 }
            );
        static readonly Matrix Expected_Sigmoid_Derivation = new Matrix
            (
                new float[] { .2139096965f, .25f, -.2287842405f, .1966119332f }
            );
        static readonly Matrix Expected_Tanh_Derivation = new Matrix
            (
                new float[] { .5590551677f, 1, .7115777626f, .4199743416f }
            );
        static readonly Matrix Expected_SoftMax_Derivation = new Matrix
            (
                new float[] { .16f, 0, -.96f, 0 }
            );
        static readonly Matrix Expected_SoftMaxwithCEL_Derivation = new Matrix
            (
                new float[] { 1, 1, 1, 1 }
            );

        #endregion

        float delta = .0001f;

        #region Testing Activations

        [TestMethod]
        public void Test_NullActivator_Activation()
        {
            var expect = Expected_NullActivator_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = NullActivator.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_ReLU_Activation()
        {
            var expect = Expected_ReLU_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = ReLU.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_LeakyReLU_Activation()
        {
            var expect = Expected_LeakyReLU_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = LeakyReLU.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_Sigmoid_Activation()
        {
            var expect = Expected_Sigmoid_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = Sigmoid.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_Tanh_Activation()
        {
            var expect = Expected_Tanh_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = Tanh.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_SoftMax_Activation()
        {
            var expect = Expected_SoftMax_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = SoftMax.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_SoftMaxWithCEL_Activation()
        {
            var expect = Expected_SoftMaxwithCEL_Activation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = SoftMaxWithCrossEntropyLoss.Activation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }

        #endregion

        #region Testing Derivations

        [TestMethod]
        public void Test_NullActivator_Derivation()
        {
            var expect = Expected_NullActivator_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = NullActivator.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_ReLU_Derivation()
        {
            var expect = Expected_ReLU_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = ReLU.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_LeakyReLU_Derivation()
        {
            var expect = Expected_LeakyReLU_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = LeakyReLU.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_Sigmoid_Derivation()
        {
            var expect = Expected_Sigmoid_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = Sigmoid.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_Tanh_Derivation()
        {
            var expect = Expected_Tanh_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = Tanh.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_SoftMax_Derivation()
        {
            var expect = Expected_SoftMax_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = SoftMax.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }
        [TestMethod]
        public void Test_SoftMaxWithCEL_Derivation()
        {
            var expect = Expected_SoftMaxwithCEL_Derivation;
            Matrix resultMatrix = new Matrix(WeightedInput.m, WeightedInput.n);
            resultMatrix = SoftMaxWithCrossEntropyLoss.Derivation(resultMatrix, WeightedInput);

            var diff = expect - resultMatrix;
            Assert.IsTrue(diff.All(x => x <= Math.Abs(delta)));
            WeightedInput.ForEach(WeightedInput_BackUp, x => x);
        }

        #endregion
    }
}
