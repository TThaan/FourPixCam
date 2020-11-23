using FourPixCam.Activators;
using MatrixHelper;
using System;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    /// <summary>
    /// Layer data.
    /// </summary>
    public class Layer
    {
        #region fields

        Func<float, float> activation, activationDerivation;

        #endregion

        #region public

        public int Id { get; internal set; }
        public int N { get; set; }
        public ActivationType ActivationType { get; set; } = ActivationType.ReLU;
        public Func<float, float> Activation => activation = default 
            ? activation = GetActivation() 
            : activation;
        public Func<float, float> ActivationDerivation => activationDerivation = default
            ? activationDerivation = GetActivationDerivation()
            : activationDerivation;
        public Matrix Weights { get; set; }
        public Matrix Biases { get; set; }

        #endregion

        public Processed Processed { get; set; }

        #region helpers

        Func<float, float> GetActivation()
        {
            switch (ActivationType)
            {
                case ActivationType.Undefined:
                    return default;
                case ActivationType.LeakyReLU:
                    return LeakyReLU.a;
                case ActivationType.NullActivator:
                    return NullActivator.a;
                case ActivationType.ReLU:
                    return ReLU.a;
                case ActivationType.Sigmoid:
                    return Sigmoid.a;
                case ActivationType.SoftMax:
                    return SoftMax.a;
                case ActivationType.Tanh:
                    return Tanh.a;
                case ActivationType.None:
                    return default;
                default:
                    return default;
            }
        }
        Func<float, float> GetActivationDerivation()
        {
            switch (ActivationType)
            {
                case ActivationType.Undefined:
                    return default;
                case ActivationType.LeakyReLU:
                    return LeakyReLU.dadz;
                case ActivationType.NullActivator:
                    return NullActivator.dadz;
                case ActivationType.ReLU:
                    return ReLU.dadz;
                case ActivationType.Sigmoid:
                    return Sigmoid.dadz;
                case ActivationType.SoftMax:
                    return SoftMax.dadz;
                case ActivationType.Tanh:
                    return Tanh.dadz;
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
    public class Processed
    {
        #region ctor & fields

        // float[] input, output, delta;
        Layer _layer;

        public Processed(Layer layer)
        {
            _layer = layer;

            Output = new Matrix(layer.N);
            Input = new Matrix(layer.N);
            Delta = new Matrix(layer.N);
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

        public void ProcessInput(Matrix input)
        {
            Input = Get_z(_layer.Weights, ReceptiveField.Processed.Input, _layer.Biases);
            Output = Get_a(_layer.Processed.Input, _layer.Activation);
            if (ProjectiveField != null)
            {
                ProjectiveField.Processed.ProcessInput(Output);
            }
            // return 
        }
        // wa: compute last layer/cost separately (maybe in NeuralNet)
        // and use 'ProcessCost(..)' only for Delta passing (not expected output)?
        public void ProcessCost(Matrix expectedOutput, Func<float, float, float> costDerivation)
        {
            if (ProjectiveField != null) 
                throw new ArgumentException("'ProcessCost(..)' can only be called in the output layer.");
            
            Delta = Get_deltaOutput(Output, expectedOutput, costDerivation, Output, _layer.ActivationDerivation);
            ReceptiveField.Processed.ProcessDelta();
            // return 
        }
        void ProcessDelta()
        {
            if (ProjectiveField == null)
                throw new ArgumentException("'ProcessDelta(..)' can only be called in a hidden layer.");
            Delta = Get_deltaHidden(ProjectiveField.Weights, ProjectiveField.Processed.Delta, Output, _layer.ActivationDerivation);
            if (ReceptiveField != null)
                ReceptiveField.Processed.ProcessDelta();
        }
    }
}
