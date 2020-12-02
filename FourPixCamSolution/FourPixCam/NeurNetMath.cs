using MatrixHelper;
using System;
using System.Linq;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    // as lib?
    internal class NeurNetMath
    {
        /// <summary>
        /// Weighted input z=wa+b.
        /// </summary>
        internal static Matrix Get_z(Matrix w, Matrix a, Matrix b)
        {
            if (b == null)
            {
                return ScalarProduct(w, a)
                    .Log($"\nZ = ");
            }
            else
            {
                return (ScalarProduct(w, a) + b)
                    .Log($"\nZ = ");
            }                
        }
        /// <summary>
        /// Activation function of the weighted input a=f(z).
        /// </summary>
        internal static Matrix Get_a(Matrix z, Func<float, float> activation)
        {
            return new Matrix(z.Select(z_j => activation(z_j)).ToArray())
                .Log($"\nA = ");
        }
        internal static Matrix Get_C(Matrix a, Matrix t, Func<float, float, float> c_j)
        {
            Matrix result = new Matrix(a.m);

            for (int j = 0; j < a.m; j++)
            {
                result[j] = c_j(a[j], t[j]);
            }

            return result
                .Log($"\n{c_j.Method.DeclaringType.Name} C =");
        }
        internal static float Get_CTotal(Matrix a, Matrix t, Func<float, float, float> c0)
        {
            // CTotal = total or averaged (i.e. sum divided by a.m)?
            return (Get_C(a, t, c0).Sum())
                .Log<float>($"\nCTotal = \n");
        }
        /// <param name="a">L</param>
        /// <param name="z">L</param>
        internal static Matrix Get_deltaOutput(
            Matrix a, Matrix t, Func<float, float, float> costDerivation,
            Matrix z, Func<float, float> activationDerivation)
        {
            Matrix result = new Matrix(a.m);

            for (int j = 0; j < a.m; j++)
            {
                result[j] = costDerivation(a[j], t[j]) * activationDerivation(z[j]);
            }

            return result
                .Log($"\ndelta =");
        }
        /// <param name="w">l+1</param>
        /// <param name="delta">l+1</param>
        /// <param name="z">l</param>
        internal static Matrix Get_deltaHidden(Matrix w, Matrix delta, Matrix z, Func<float, float> dadzFunction)
        {
            Matrix dCda = w.Transpose * delta;
            Matrix dadz = new Matrix(z.Select(x => dadzFunction(x)).ToArray());
            
            return HadamardProduct(dCda, dadz)
                .Log($"\ndelta =");
        }
        /// <param name="w">l</param>
        /// <param name="delta">l</param>
        /// <param name="a">l-1</param>
        internal static Matrix Get_CorrectedWeights(Matrix w, Matrix delta, Matrix a, float learningRate)
        {
            Matrix result = new Matrix(w.m, w.n);
            result = w - learningRate * (delta * a.Transpose);
            return result
                .Log($"\nnextW =");
        }
        /// <param name="b">l</param>
        /// <param name="delta">l</param>
        internal static Matrix Get_CorrectedBiases(Matrix b, Matrix delta, float learningRate)
        {
            return (b - learningRate * delta)
                .Log($"\nnextB =");
        }
    }
}
