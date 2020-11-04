using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    // wa: not static but as idisp-instance for each back-prop
    // including stored arrays '..ofLayer[l]'?

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
                    return LeakyReLU.a(z);
                case ActivationType.NullActivator:
                    return NullActivator.a(z);
                case ActivationType.Sigmoid:
                    return Sigmoid.a(z);
                case ActivationType.SoftMax:
                    return SoftMax.a(z);
                default:
                    throw new System.ArgumentException("Undefined activator function");
            }
        }
        public static Matrix dadz(Matrix z, ActivationType activationType)
        {
            switch (activationType)
            {
                case ActivationType.Undefined:
                    throw new ArgumentException("Undefined activator function");
                case ActivationType.LeakyReLU:
                    return LeakyReLU.dadz(z);
                case ActivationType.NullActivator:
                    return NullActivator.dadz(z);
                case ActivationType.Sigmoid:
                    return Sigmoid.dadz(z);
                case ActivationType.SoftMax:
                    return SoftMax.dadz(z);
                default:
                    throw new ArgumentException("Undefined activator function");
            }
        }
        public static float C(Matrix a, Matrix t, CostType costType)
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
        public static Matrix dCda(Matrix a, Matrix t, CostType costType)
        {
            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    return SquaredMeanError.dCda(a, t);
                default:
                    throw new ArgumentException("Undefined cost function");
            }
        }
        public static Matrix deltaOfOutputLayer(Matrix a, Matrix t, CostType costType, Matrix dadz)
        {
            return HadamardProduct(dCda(a, t, costType), dadz);
        }
        public static Matrix deltaOfHiddenLayer(Matrix w, Matrix error, Matrix dadz)
        {
            Matrix wTranspose = w.GetTranspose();
            // rename tmp
            Matrix tmp = ScalarProduct(wTranspose, error);
            return HadamardProduct(tmp, dadz);
        }
    }
}
