using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using MatrixHelper;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    public class LearningNet
    {
        #region fields

        // readonly Random rnd = RandomProvider.GetThreadRandom();        
        NeuralNet net;

        #endregion

        #region ctor

        public LearningNet(NeuralNet net)      // param jasonFile
        {
            this.net = net;

            CostType = CostType.SquaredMeanError;
            ActivationType_OfLayer = GetActivationTypes();

            a_OfLayer = new Matrix[net.L];
            z_OfLayer = new Matrix[net.L];
        }

        #region helper methods

        ActivationType[] GetActivationTypes()
        {
            // skip activator for first "layer"
            return Enumerable.Range(2, net.L)
                .Select(x => ActivationType.LeakyReLU)
                .ToArray();
        }

        #endregion

        #endregion

        #region properties

        // public Matrix x { get; set; }

        /// <summary>
        /// expected output
        /// </summary>
        public Matrix t { get; set; }
        public Matrix C0 { get; set; }
        /// <summary>
        /// total value (= wa + b)
        /// </summary>
        public Matrix[] z_OfLayer { get; set; }
        /// <summary>
        /// activation (= f(z))
        /// </summary>
        public Matrix[] a_OfLayer { get; set; }
        public Matrix[] error_OfLayer { get; set; }
        public Matrix[] f_OfLayer { get; set; } // => activations[]
        public Matrix[] dfda_OfLayer { get; set; }  // => Matrix.Partial(f, a);

        public CostType CostType { get; set; }
        public float LastCost { get; set; }
        ActivationType[] ActivationType_OfLayer;


        #endregion

        #region methods

        public void FeedForward(Matrix input)
        {
            // a[0] = input layer
            a_OfLayer[0] = new Matrix(input.ToArray());

            // iterate over layers (skip input layer)
            for (int i = 1; i < net.L; i++)
            {
                z_OfLayer[i] = NeurNetMath.z(net.w[i], a_OfLayer[i - 1], net.b[i]);
                a_OfLayer[i] = NeurNetMath.a(z_OfLayer[i], ActivationType_OfLayer[i]);
                // z[i] = Operations.ScalarProduct(net.w[i], a[i - 1]) + net.b[i];
                // a[i] = activators[i].GetValue(z[i]);
            }
        }
        public Matrix GetTotalCostOfLastSample(Matrix t)
        {
            return C0 = NeurNetMath.C0(a_OfLayer.Last(), t, CostType.SquaredMeanError); // CostType.C(a.Last(), t);
        }
        public void BackPropagate(Matrix t) // wa Matrix C0 as parameter..
        {
            error_OfLayer = new Matrix[net.L]; // Matrix[] result

            for (int i = net.L - 1; i >= 0; i--)
            {
                Matrix delta;

                if (i == net.L - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    delta = NeurNetMath.delta(a_OfLayer[i], t, CostType.SquaredMeanError, ActivationType_OfLayer[i], z_OfLayer[i]);
                }
                else
                {
                    delta = Operations.ScalarProduct(net.w[i + 1].GetTranspose(), error_OfLayer[i + 1]) * LeakyReLU.df(z_OfLayer[i]);// NeurNetMath.delta(null, null, default, default, null); //   //delta = layer.GetDeltaOfH(Net.Layers[i + 1].d);
                }

                error_OfLayer[i] = delta;
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
