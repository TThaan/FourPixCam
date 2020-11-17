using MatrixHelper;
using System;
using System.Linq;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    // Exchange activation type parameters with funcs!?

    // wa: not static but as idisp-instance for each back-prop
    // including stored arrays '..ofLayer[l]'?

    public class NeurNetMath
    {
        // redundant?
        public enum CostType
        {
            Undefined, SquaredMeanError
        }

        /// <summary>
        /// Weighted input z=wa+b.
        /// </summary>
        public static Matrix Get_z(Matrix w, Matrix a, Matrix b)
        {
            return ScalarProduct(w, a) + b;
        }
        /// <summary>
        /// Activation function of the weighted input a=f(z).
        /// </summary>
        public static Matrix Get_a(Matrix z, Func<float, float> activation)
        {
            return new Matrix(
                z.Select(z_j => activation(z_j)).ToArray()
                );  // ToMatrix()?
        }
        /// <summary>
        /// Partial derivation of a with respect to z.
        /// </summary>
        public static Matrix Get_dadz(Matrix a, Matrix z, Func<float, float, float> derivationOfActivation)
        {
            return Partial(a, z, derivationOfActivation);
        }
        public static Matrix Get_C(Matrix a, Matrix t, Func<float, float, float> c_j)
        {
            Matrix result = new Matrix(a.m);

            for (int j = 0; j < a.m; j++)
            {
                result[j] = c_j(a[j], t[j]);
            }

            return result;
        }
        public static float Get_CTotal(Matrix a, Matrix t, Func<float, float, float> c0)
        {
            // CTotal = total or averaged (i.e. sum divided by a.m)?
            return Get_C(a, t, c0).Sum();
        }
        /// <summary>
        /// = delta * w
        /// </summary>
        /// <returns></returns>
        public static Matrix Get_dCda(float C, Matrix a, Func<float, float, float> derivationOfActivation)
        {
            Matrix result = a.Transpose;

            for (int j = 0; j < a.m; j++)
            {
                result[1] = derivationOfActivation(C, a[1]);
            }

            return result;
        }
        /// <param name="a">L</param>
        /// <param name="z">L</param>
        public static Matrix Get_deltaOutput(
            Matrix a, Matrix t, Func<float, float, float> costDerivation, 
            Matrix z, Func<float, float> activationDerivation)
        {
            Matrix result = new Matrix(a.m);

            for (int j = 0; j < a.m; j++)
            {
                result[j] = costDerivation(a[j], t[j]) * activationDerivation(z[j]);
            }

            return result;
        }
        /// <param name="w">l+1</param>
        /// <param name="delta">l+1</param>
        /// <param name="z">l</param>
        public static Matrix Get_deltaHidden(Matrix w, Matrix delta, Matrix z, Func<float, float> dadzFunction)
        {
            Matrix dCda = w.Transpose * delta;
            Matrix dadz = new Matrix(z.Select(x => dadzFunction(x)).ToArray());
            
            return HadamardProduct(dCda, dadz);
        }
        /// <param name="w">l</param>
        /// <param name="delta">l</param>
        /// <param name="a">l-1</param>
        public static Matrix Get_CorrectedWeights(Matrix w, Matrix delta, Matrix a, float learningRate)
        {
            return w - learningRate * (delta*a.Transpose);
        }
        /// <param name="b">l</param>
        /// <param name="delta">l</param>
        public static Matrix Get_CorrectedBiases(Matrix b, Matrix delta, float learningRate)
        {
            return b - learningRate * delta;
        }
    }
}
