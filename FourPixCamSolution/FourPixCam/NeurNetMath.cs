using FourPixCam.Activators;
using FourPixCam.CostFunctions;
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
        //internal static void Set_z(Matrix result, Matrix w, Matrix a, Matrix b)
        //{
        //    if (b == null)
        //    {
        //        result.GetScalarProduct(w, a);
        //    }
        //    else
        //    {
        //        result.GetScalarProduct(w, a).Add(b);
        //    }                
        //}
        /// <summary>
        /// Activation function of the weighted input a=f(z).
        /// </summary>
        //internal static void Set_a(Matrix z, Func<Matrix, Matrix> activation)
        //{
        //    activation(z);
        //    // Bad workaround!
        //    // Find better solution, e.g. activation = Func<Matrix, float)
        //    // return new Matrix(z.Select(z_j => activation(z_j)).ToArray())
        //    //     .Log($"\nA = ");
        //}
        //internal static Matrix Get_C(Matrix a, Matrix t, Func<float, float, float> c_j)
        //{
        //    Matrix result = new Matrix(a.m);

        //    for (int j = 0; j < a.m; j++)
        //    {
        //        result[j] = c_j(a[j], t[j]);
        //    }

        //    return result
        //        .Log($"\n{c_j.Method.DeclaringType.Name} C =");
        //}
        //internal static float Get_CTotal(Matrix a, Matrix t, Func<float, float, float> c0)
        //{
        //    // CTotal = total or averaged (i.e. sum divided by a.m)?
        //    return (Get_C(a, t, c0).Sum())
        //        .Log<float>($"\nCTotal = \n");
        //}
        /// <param name="a">L</param>
        /// <param name="z">L</param>
        internal static void Set_deltaOutput(Matrix result, 
            Matrix a, Matrix t, Func<Matrix, Matrix, Matrix> costDerivation,
            Matrix z, Func<Matrix, Matrix> activationDerivation)
        {
            // Changes a to costDerivation
            // costDerivation(a, t);

            // Changes z to activationDerivation
            // activationDerivation(z);

            // Changes costDerivation (former a) to delta of output layer
            result = costDerivation(a, t).GetHadamardProduct(activationDerivation(z));
        }
        /// <param name="w">l+1</param>
        /// <param name="delta">l+1</param>
        /// <param name="z">l</param>
        //internal static Matrix Get_deltaHidden(Matrix w, Matrix delta, Matrix z, Func<Matrix, Matrix> activationDerivation)
        //{
        //    Matrix dCda = w.Transpose * delta;
        //    Matrix dadz = activationDerivation(z);
            
        //    return HadamardProduct(dCda, dadz)
        //        .Log($"\ndelta =");
        //}
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
