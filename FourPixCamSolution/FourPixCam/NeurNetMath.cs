using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    public class NeurNetMath
    {
        public enum ActivationType
        {
            Undefined, LeakyReLU, NullActivator, Sigmoid, SoftMax
        }
        public enum CostType
        {
            Undefined, SquaredMeanError
        }

        public static Matrix z(Matrix w, Matrix a, Matrix b)
        {
            return ScalarProduct(w, a) + b;
        }
        public static Matrix a(Matrix z, ActivationType activationType)
        {
            switch (activationType)
            {
                case ActivationType.Undefined:
                    throw new System.ArgumentException("Undefined activator function");
                case ActivationType.LeakyReLU:
                    return LeakyReLU.f(z);
                case ActivationType.NullActivator:
                    return NullActivator.f(z);
                case ActivationType.Sigmoid:
                    return Sigmoid.f(z);
                case ActivationType.SoftMax:
                    return SoftMax.f(z);
                default:
                    throw new System.ArgumentException("Undefined activator function");
            }
        }
        public static Matrix delta(Matrix a, Matrix t, CostType costType, ActivationType activationType, Matrix z)
        {
            Matrix cost;

            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    cost = C0(a, t, costType);
                    break;
                default:
                    throw new ArgumentException("Undefined cost function");
            }

            switch (activationType)
            {
                case ActivationType.Undefined:
                    throw new System.ArgumentException("Undefined activator function");
                case ActivationType.LeakyReLU:
                    return cost * LeakyReLU.df(z);
                case ActivationType.NullActivator:
                    return cost * NullActivator.df(z);
                case ActivationType.Sigmoid:
                    return cost * Sigmoid.df(z);
                case ActivationType.SoftMax:
                    return cost * SoftMax.df(z);
                default:
                    throw new System.ArgumentException("Undefined activator function");
            }
        }
        /// <summary>
        /// Return matrix or scalar?
        /// </summary>
        public static Matrix C0(Matrix a, Matrix t, CostType costType)
        {
            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    return SquaredMeanError.C(a, t);
                default:
                    throw new ArgumentException("Undefined cost function");
            }
        }
    }
}
