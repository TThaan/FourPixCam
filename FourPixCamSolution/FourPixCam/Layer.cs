using FourPixCam.Activators;
using MatrixHelper;
using NNet_InputProvider;
using System;
using static FourPixCam.NeurNetMath;
using static MatrixHelper.Operations2;

namespace FourPixCam
{
    /// <summary>
    /// Layer data.
    /// </summary>
    [Serializable]
    public class Layer
    {
        #region ctor & fields

        public Layer(int id, int n, ActivationType activationType)
        {
            Id = id;
            N = n;
            ActivationType = activationType;
            Activation = GetActivation();
            ActivationDerivation = GetActivationDerivation();
            Processed = new Processed(this);
        }

        #endregion

        #region public

        public int Id { get; set; }
        public int N { get; set; }
        public ActivationType ActivationType { get; set; }
        // Better define as Inteface/Abstract:
        public Func<Matrix, Matrix, Matrix> Activation { get; set; }
        // Better define as Inteface/Abstract:
        public Func<Matrix, Matrix, Matrix> ActivationDerivation { get; set; }
        public Matrix Weights { get; set; }
        public Matrix Biases { get; set; }

        public Processed Processed { get; set; }    // as child class?

        #endregion

        #region helpers

        // in factory?
        Func<Matrix, Matrix, Matrix> GetActivation()
        {
            switch (ActivationType)
            {
                case ActivationType.Undefined:
                    return default;
                case ActivationType.LeakyReLU:
                    return LeakyReLU.Activation;
                case ActivationType.NullActivator:
                    return NullActivator.Activation;
                case ActivationType.ReLU:
                    return ReLU.Activation;
                case ActivationType.Sigmoid:
                    return Sigmoid.Activation;
                case ActivationType.SoftMax:
                    return SoftMax.Activation;
                case ActivationType.SoftMaxWithCrossEntropyLoss:
                    return SoftMaxWithCrossEntropyLoss.Activation;
                case ActivationType.Tanh:
                    return Tanh.Activation;
                case ActivationType.None:
                    return default;
                default:
                    return default;
            }
        }
        Func<Matrix, Matrix, Matrix> GetActivationDerivation()
        {
            switch (ActivationType)
            {
                case ActivationType.Undefined:
                    return default;
                case ActivationType.LeakyReLU:
                    return LeakyReLU.Derivation;
                case ActivationType.NullActivator:
                    return NullActivator.Derivation;
                case ActivationType.ReLU:
                    return ReLU.Derivation;
                case ActivationType.Sigmoid:
                    return Sigmoid.Derivation;
                case ActivationType.SoftMax:
                    return SoftMax.Derivation;
                case ActivationType.SoftMaxWithCrossEntropyLoss:
                    return SoftMaxWithCrossEntropyLoss.Derivation;
                case ActivationType.Tanh:
                    return Tanh.Derivation;
                case ActivationType.None:
                    return default;
                default:
                    return default;
            }
        }

        #endregion
    }

    /// <summary>
    /// Layer logic & fluent data.
    /// </summary>
    [Serializable]
    public class Processed  // : IDisposable
    {
        #region ctor & fields

        Layer _layer;

        public Processed(Layer layer)
        {
            _layer = layer;
            Reset();
        }

        #endregion

        /// <summary>
        /// "Weighted Sum" (z).
        /// </summary>
        public Matrix Input { get; set; }
        /// <summary>
        /// "Activated output" (a).
        /// </summary>
        public Matrix Output { get; set; }
        /// <summary>
        /// Bad naming, DCDA = derivation of the cost with respect to this layers' output
        /// in output layer: DCDA = costDerivation(DCDA, expectedOutput)
        /// in hidden layer: DCDA = ScalarProduct(ProjectiveField.Weights.Transpose, ProjectiveField.Processed.Delta)
        /// </summary>
        public Matrix DCDA { get; set; }
        public Matrix DADZ { get; set; }
        public Matrix Delta { get; set; }
        /// <summary>
        /// 'Previous' Layer (providing input to this layer,
        /// i.e. 'perceived' by this layer).
        /// </summary>
        public Layer ReceptiveField { get; set; }
        /// <summary>
        /// 'Following' Layer (receiving input from this layer,
        /// i.e. 'projected (to)' by this layer).
        /// </summary>
        public Layer ProjectiveField { get; set; }

        public void Reset()
        {
            Output = new Matrix(_layer.N);
            Input = new Matrix(_layer.N);
            DCDA = new Matrix(_layer.N);
            DADZ = new Matrix(_layer.N);
            Delta = new Matrix(_layer.N);
        }
        public void ProcessInput(Matrix originalInput = null)
        {
            SetInput(originalInput);
            if (_layer.Id > 0)
            {
                Input.Log($"\n\nInput L{_layer.Id}");
            }
            SetOutput();
            if (_layer.Id > 0)
            {
                Output.Log($"\n\nOutput L{_layer.Id}");
            }
            ProjectiveField?.Processed.ProcessInput();
        }
        public void ProcessDelta(Matrix expectedOutput, Action<Matrix, Matrix, Matrix> costDerivation)
        {
            SetDCDA(Output, expectedOutput, costDerivation);
            DCDA.Log($"\n\nDCDA L{_layer.Id}");

            // Test:

            // SetDCDA(Output, expectedOutput, costFunction);

            //var dcdaMatrix = new Matrix(DCDA.m, DCDA.n);
            //for (int j = 0; j < dcdaMatrix.m; j++)
            //{
            //    float dcdaFloat = (debugSample.ExpectedOutput[j] - Output[j]) * (debugSample.ExpectedOutput[j] - Output[j]);
            //    dcdaMatrix[j] = dcdaFloat;
            //}

            SetDADZ();
            DADZ.Log($"\n\nDADZ L{_layer.Id}");
            SetDelta();
            Delta.Log($"\n\nDelta L{_layer.Id}");
            ReceptiveField?.Processed.ProcessDelta(expectedOutput, costDerivation);
        }

        #region helpers

        public void SetInput(Matrix originalInput)
        {
            if (originalInput != null)
            {
                Input = new Matrix(originalInput);
            }
            else
            {
                // Input.ForEach(x => 0);
                SetScalarProduct(_layer.Weights, ReceptiveField.Processed.Output, Input);
            }

            if (_layer.Biases != null)
                Input = Input.Add(_layer.Biases);
        }
        //public void SetInput(Matrix originalInput)
        //{
        //    if (originalInput != null)
        //    {
        //        Input = new Matrix(originalInput);
        //    }
        //    else
        //    {
        //        // Input.ForEach(x => 0);
        //        SetScalarProduct(_layer.Weights, ReceptiveField.Processed.Output, Input);
        //    }

        //    if (_layer.Biases != null)
        //        Input = Input.Add(_layer.Biases);
        //}
        public void SetOutput()
        {
            Output.ForEach(x => 0);
            Output = _layer.Activation(Output, Input);
        }
        public void SetDCDA(Matrix actualOutput, Matrix expectedOutput, Action<Matrix, Matrix, Matrix> setCostDerivation)
        {
            DCDA.ForEach(x => 0);

            if (ProjectiveField == null)
            {
                setCostDerivation(actualOutput, expectedOutput, DCDA);
            }
            else
            {
                SetScalarProduct(ProjectiveField.Weights.GetTranspose(), ProjectiveField.Processed.Delta, DCDA);
            }
        }
        public void SetDADZ()
        {
            DADZ.ForEach(x => 0);
            DADZ = _layer.ActivationDerivation(DADZ, Input);
        }
        public void SetDelta()
        {
            Delta.ForEach(x => 0);
            SetHadamardProduct(DCDA, DADZ, Delta);
        }

        public void AdaptWeightsAndBiases(float learningRate)
        {
            if (ReceptiveField != null)
            {
                Matrix tmp1 = new Matrix(Delta.m, _layer.Processed.ReceptiveField.Processed.Output.m);
                Matrix weightsChange = new Matrix(Delta.m, _layer.Processed.ReceptiveField.Processed.Output.m);
                Matrix newWeights = new Matrix(Delta.m, _layer.Processed.ReceptiveField.Processed.Output.m);

                SetScalarProduct(Delta, _layer.Processed.ReceptiveField.Processed.Output.GetTranspose(), tmp1);
                Multiplicate(tmp1, -learningRate, weightsChange);
                Add(_layer.Weights, weightsChange, newWeights);

                _layer.Weights.Log($"\n\nOldW L{_layer.Id}");
                weightsChange.Log($"\n\ndW (variant 1)  L{_layer.Id}");
                Matrix diff = (newWeights - _layer.Weights).Log($"\n\ndW (variant 2) L{_layer.Id}");
                newWeights.Log($"\n\nNextW L{_layer.Id}");

                _layer.Weights = new Matrix(newWeights);
                // _layer.Weights = Get_CorrectedWeights(_layer.Weights, Delta, _layer.Processed.ReceptiveField.Processed.Output, learningRate);
            }
            
            if (_layer.Biases != null)
            {
                // _layer.Biases = Get_CorrectedBiases(_layer.Biases, Delta, learningRate);
            }
        }
        
        #endregion
    }
}
