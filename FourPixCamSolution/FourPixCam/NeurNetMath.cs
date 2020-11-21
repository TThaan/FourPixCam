using MatrixHelper;
using System;
using System.Linq;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    /// <summary>
    /// Dumping as optional param (only) here?
    /// wa: ILogger injection?
    /// </summary>
    public class NeurNetMath
    {
        public static bool IsLogOn { get; set; }

        /// <summary>
        /// Weighted input z=wa+b.
        /// </summary>
        public static Matrix Get_z(Matrix w, Matrix a, Matrix b)
        {
            return (ScalarProduct(w, a) + b)
                .LogIt($"\nZ = ");
        }
        /// <summary>
        /// Activation function of the weighted input a=f(z).
        /// </summary>
        public static Matrix Get_a(Matrix z, Func<float, float> activation)
        {
            return new Matrix(z.Select(z_j => activation(z_j)).ToArray())
                .LogIt($"\nA = ");
        }
        public static Matrix Get_C(Matrix a, Matrix t, Func<float, float, float> c_j)
        {
            Matrix result = new Matrix(a.m);

            for (int j = 0; j < a.m; j++)
            {
                result[j] = c_j(a[j], t[j]);
            }

            return result
                .LogIt($"\n{c_j.Method.DeclaringType.Name} C =");
        }
        public static float Get_CTotal(Matrix a, Matrix t, Func<float, float, float> c0)
        {
            // CTotal = total or averaged (i.e. sum divided by a.m)?
            return (Get_C(a, t, c0).Sum())
                .LogIt($"\nCTotal = \n");
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

            return result
                .LogIt($"\ndelta =");
        }
        /// <param name="w">l+1</param>
        /// <param name="delta">l+1</param>
        /// <param name="z">l</param>
        public static Matrix Get_deltaHidden(Matrix w, Matrix delta, Matrix z, Func<float, float> dadzFunction)
        {
            Matrix dCda = w.Transpose * delta;
            Matrix dadz = new Matrix(z.Select(x => dadzFunction(x)).ToArray());
            
            return HadamardProduct(dCda, dadz)
                .LogIt($"\ndelta =");
        }
        /// <param name="w">l</param>
        /// <param name="delta">l</param>
        /// <param name="a">l-1</param>
        public static Matrix Get_CorrectedWeights(Matrix w, Matrix delta, Matrix a, float learningRate)
        {
            return w - learningRate * (delta*a.Transpose)
                .LogIt($"\nnextW =");
        }
        /// <param name="b">l</param>
        /// <param name="delta">l</param>
        public static Matrix Get_CorrectedBiases(Matrix b, Matrix delta, float learningRate)
        {
            return (b - learningRate * delta)
                .LogIt($"\nnextB =");
        }
    }
}
