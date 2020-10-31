using _4PixCam.Activators;
using _4PixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Linq;
using Activator = _4PixCam.Activators.Activator;

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

            CostFunction = new SquaredMeanError();
            activators = GetActivators();

            a = new Matrix[net.Length];
            z = new Matrix[net.Length];
        }

        #region helper methods

        Activator[] GetActivators()
        {

            // skip activator for first "layer"
            return Enumerable.Range(2, net.Length)
                .Select(x => new LeakyReLU())
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
        public Matrix[] z { get; set; }
        /// <summary>
        /// activation (= f(z))
        /// </summary>
        public Matrix[] a { get; set; }
        public Matrix[] d { get; set; }
        public Matrix[] f { get; set; } // => activations[]
        public Matrix[] dfda { get; set; }  // => Matrix.Partial(f, a);

        public CostFunction CostFunction { get; set; }
        public float LastCost { get; set; }
        Activator[] activators;


        #endregion

        #region methods

        public void FeedForward(Matrix x)
        {
            a[0] = new Matrix(x.ToArray());

            // iterate over layers (skip input layer)
            for (int i = 1; i < net.Length; i++)
            {
                z[i] = Operations.ScalarProduct(net.w[i], a[i - 1]) + net.b[i];
                a[i] = activators[i].GetValue(z[i]);
            }
        }
        public Matrix GetTotalCostOfLastSample(Matrix t)
        {
            return C0 = CostFunction.GetCost(a.Last(), t);
        }
        public void BackPropagate(Matrix t) // wa Matrix C0 as parameter..
        {
            d = new Matrix[net.Length]; // Matrix[] result

            for (int i = net.Length - 1; i >= 0; i--)
            {
                Matrix delta;

                if (i == net.Length - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    delta = CostFunction.GetCostDerivative(a[i], t) * activators[i].GetDerivativeWithRespectTo(z[i]);   //layer.GetDeltaOfL(t, Net.CostFunction);
                }
                else
                {
                    delta = Operations.ScalarProduct(net.w[i+1].GetTranspose(), d[i+1]) * activators[i].GetDerivativeWithRespectTo(z[i]);   //delta = layer.GetDeltaOfH(Net.Layers[i + 1].d);
                }

                d[i] = delta;
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
