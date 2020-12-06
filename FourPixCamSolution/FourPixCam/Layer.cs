using FourPixCam.Activators;
using MatrixHelper;
using System;
using static FourPixCam.NeurNetMath;

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
        public Func<Matrix, Matrix, Matrix> Activation { get; set; }
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

        public void Reset()
        {
            Output = new Matrix(_layer.N);
            Input = new Matrix(_layer.N);
            DCDA = new Matrix(_layer.N);
            Delta = new Matrix(_layer.N);
        }

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

        // Wa: diff weight/bias range for diff layer?

        public void ProcessInput(Matrix originalInput = null)
        {
            // Test: Compare with and without return value (get vs set).
            // Use ExtMeth in Activators?

            if (originalInput != null)
            {
                Input = new Matrix(originalInput);
            }
            else
            {
                Input.ForEach(x => 0);
                Input = Input.GetScalarProduct(_layer.Weights, ReceptiveField.Processed.Output);
            }

            if (_layer.Biases != null)
                Input = Input.Add(_layer.Biases);
            Output.ForEach(x => 0);
            Output = _layer.Activation(Output, Input);

            ProjectiveField?.Processed.ProcessInput();
        }
        public void ProcessOutputDelta(Matrix expectedOutput, Func<Matrix, Matrix, Matrix> costDerivation, float learningRate)
        {
            if (ProjectiveField != null) 
                throw new ArgumentException("'ProcessCost(..)' can only be called in the output layer.");

            DCDA.ForEach(x => 0);
            DCDA = costDerivation(DCDA, expectedOutput);

            Delta.ForEach(x => 0);
            Delta = _layer.ActivationDerivation(Delta, Input).GetHadamardProduct(DCDA);
            AdaptWeightsAndBiases(learningRate);
            ReceptiveField?.Processed.ProcessHiddenDelta(learningRate);
        }

        #region helpers

        void ProcessHiddenDelta(float learningRate)
        {
            if (ProjectiveField == null)
                throw new ArgumentException("'ProcessDelta(..)' can only be called in a hidden layer.");

            DCDA.ForEach(x => 0);
            DCDA = DCDA.GetScalarProduct(ProjectiveField.Weights.Transpose, ProjectiveField.Processed.Delta);

            Delta.ForEach(x => 0);
            Delta = _layer.ActivationDerivation(Delta, Input).GetHadamardProduct(DCDA);
            AdaptWeightsAndBiases(learningRate);
            ReceptiveField?.Processed.ProcessHiddenDelta(learningRate);
        }
        void AdaptWeightsAndBiases(float learningRate)
        {
            if (ReceptiveField != null)
            {
                _layer.Weights = Get_CorrectedWeights(_layer.Weights, Delta, _layer.Processed.ReceptiveField.Processed.Output, learningRate);
            }
            
            if (_layer.Biases != null)
            {
                _layer.Biases = Get_CorrectedBiases(_layer.Biases, Delta, learningRate);
            }
        }
        
        #endregion
    }
}
