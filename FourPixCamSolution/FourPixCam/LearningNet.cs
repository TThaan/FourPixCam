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

            z_OfLayer = new Matrix[net.L];
            a_OfLayer = new Matrix[net.L];
            dadz_OfLayer = new Matrix[net.L];
            error_OfLayer = new Matrix[net.L];
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
        public Matrix t { get; set; }   // redundant?
        public float C { get; set; }
        /// <summary>
        /// total value (= wa + b)
        /// </summary>
        public Matrix[] z_OfLayer { get; set; }
        /// <summary>
        /// activation (= f(z))
        /// </summary>
        public Matrix[] a_OfLayer { get; set; }
        public Matrix[] dadz_OfLayer { get; set; }  // => Matrix.Partial(f, a);
        public Matrix[] error_OfLayer { get; set; }
        public Matrix[] f_OfLayer { get; set; } // => activations[]

        public CostType CostType { get; set; }
        public float LastCost { get; set; } // redundant?
        ActivationType[] ActivationType_OfLayer;


        #endregion

        #region methods

        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            // a[0] = input layer
            a_OfLayer[0] = new Matrix(input.ToArray());

            // iterate over layers (skip input layer)
            for (int i = 1; i < net.L; i++)
            {
                z_OfLayer[i] = NeurNetMath.z(net.w[i], a_OfLayer[i - 1], net.b[i]);
                a_OfLayer[i] = NeurNetMath.a(z_OfLayer[i], ActivationType_OfLayer[i]);
            }

            return a_OfLayer.Last();
        }
        public void BackPropagate(Matrix y)
        {
            // debug
            // var v1 = NeurNetMath.C(a_OfLayer[net.L-1], y, CostType.SquaredMeanError);

            // Iterate backwards over each layer (skip input layer).
            for (int l = net.L - 1; l > 0; l--)
            {
                dadz_OfLayer[l] = NeurNetMath.dadz(z_OfLayer[l], ActivationType_OfLayer[l]);
                Matrix error;

                if (l == net.L - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    error = NeurNetMath.deltaOfOutputLayer(a_OfLayer[l], y, CostType.SquaredMeanError, dadz_OfLayer[l]);
                }
                else
                {
                    error = NeurNetMath.deltaOfHiddenLayer(net.w[l + 1], error_OfLayer[l + 1], dadz_OfLayer[l]);
                }

                error_OfLayer[l] = error;
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
