using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using FourPixCam.Enums;
using MatrixHelper;
using System;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    // wa: not static but as idisp-instance for each back-prop
    // including stored arrays '..ofLayer[l]'?

    public class NeurNetMath
    {
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
                    throw new ArgumentException("Undefined activator function");
                case ActivationType.LeakyReLU:
                    return LeakyReLU.a(z);
                case ActivationType.NullActivator:
                    return NullActivator.a(z);
                case ActivationType.ReLU:
                    return ReLU.a(z);
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
                case ActivationType.ReLU:
                    return ReLU.dadz(z);
                case ActivationType.Sigmoid:
                    return Sigmoid.dadz(z);
                case ActivationType.SoftMax:
                    return SoftMax.dadz(z);
                default:
                    throw new ArgumentException("Undefined activator function");
            }
        }
        public static Matrix C(Matrix a, Matrix t, CostType costType)
        {
            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    return SquaredMeanError.E(a, t);
                default:
                    throw new ArgumentException("Undefined cost function");
            }
        }
        public static float CTotal(Matrix a, Matrix t, CostType costType)
        {
            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    return SquaredMeanError.ETotal(a, t);
                default:
                    throw new ArgumentException("Undefined cost function");
            }
        }
        public static float dCda(float a, float t, CostType costType)
        {
            switch (costType)
            {
                case CostType.Undefined:
                    throw new ArgumentException("Undefined cost function");
                case CostType.SquaredMeanError:
                    return SquaredMeanError.dEda(a, t);
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
                    return SquaredMeanError.dEda(a, t);
                default:
                    throw new ArgumentException("Undefined cost function");
            }
        }
        public static Matrix deltaOfOutputLayer(Matrix a, Matrix t, CostType costType, Matrix dadz)
        {
            Matrix result = new Matrix(a.m, 1);

            for (int j = 0; j < a.m; j++)
            {
                result[j, 0] = dCda(a[j, 0], t[j, 0], costType) * dadz[j, 0];
            }
            var check = HadamardProduct(dCda(a, t, costType), dadz);

            return result;
        }
        //public static Matrix deltaOfOutputLayer(Matrix a, Matrix t, CostType costType, Matrix dadz)
        //{
        //    return HadamardProduct(dCda(a, t, costType), dadz);
        //}
        public static Matrix deltaOfHiddenLayer(Matrix w, Matrix error, Matrix dadz)
        {
            Matrix wTranspose = w.GetTranspose();
            // rename tmp
            Matrix tmp = ScalarProduct(wTranspose, error);
            return HadamardProduct(tmp, dadz);
        }
        public static Matrix GetCorrectedWeights(Matrix w, Matrix a, Matrix e, float learningRate)
        {
            // = dCda*dadz*dzdw = error^L * a^(L-1) ??


            // Matrix dCdw = HadamardProduct(e, a);
            // return w - learningRate * dCdw;

            Matrix result = new Matrix(w.m, w.n);
            for (int j = 0; j < w.m; j++)
            {
                for (int k = 0; k < w.n; k++)
                {
                    result[j, k] = w[j, k] - a[k, 0] * e[j, 0];
                }
            }

            return result;
        }
        public static Matrix GetCorrectedBiases(Matrix b, Matrix e, float learningRate)
        {
            return b - learningRate * e;
        }
    }
}
