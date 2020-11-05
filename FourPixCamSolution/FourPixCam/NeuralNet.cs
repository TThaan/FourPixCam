using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam
{
    public class NeuralNet
    {
        #region fields

        readonly Random rnd = RandomProvider.GetThreadRandom();
        
        #endregion

        #region ctor

        public NeuralNet(params int[] layers)      // param jasonFile
        {
            NeuronsPerLayer = layers;
            L = layers.Count();

            WeightsRange = 2;
            BiasRange = 5;

            w = GetWeights();
            b = GetBiases();

            //w = new[]
            //{
            //    null, 
            //    new Matrix(new [,]
            //    {
            //        { .4f, .8f, -.4f},
            //        { .6f, .8f, -.6f},
            //        { .4f, .6f, -.4f},
            //        { .8f, .2f, -.2f }
            //    }),
            //    new Matrix(new [,]
            //    {
            //        { .1f, -.1f, .2f, -.2f},
            //        { .2f, -.2f, .4f, -.4f}
            //    })
            //};

            //b = new[]
            //{
            //    null,
            //    new Matrix(new [,]
            //    {
            //        { 1f },
            //        { 3f },
            //        { 2f },
            //        { 2f }
            //    }),
            //    new Matrix(new [,]
            //    {
            //        { 2f },
            //        { 4f }
            //    })
            //};
        }

        #region helper methods

        Matrix[] GetWeights()
        {
            Matrix[] result = new Matrix[L];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < L; l++)
            {
                Matrix weightsOfThisLayer = new Matrix(NeuronsPerLayer[l], NeuronsPerLayer[l - 1]);

                for (int j = 0; j < NeuronsPerLayer[l]; j++)
                {
                    for (int k = 0; k < NeuronsPerLayer[l-1]; k++)
                    {
                        // The entry in the j-th row and k-th colum is w^l_jk
                        // i.e. the weight connecting
                        // from the k-th neuron of layer l-1
                        // to the jth neuron of layer l.
                        weightsOfThisLayer[j, k] = WeightsRange / 2 * GetSmallRandomNumber();
                    }
                };

                result[l] = weightsOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        Matrix[] GetBiases()
        {
            Matrix[] result = new Matrix[L];

            // Iterate over layers (skip first layer).
            for (int l = 1; l < L; l++)
            {
                Matrix biasesOfThisLayer = new Matrix(NeuronsPerLayer[l], 1);

                for (int j = 0; j < NeuronsPerLayer[l]; j++)
                {
                    biasesOfThisLayer[j, 0] = BiasRange / 2 * GetSmallRandomNumber();
                };

                result[l] = biasesOfThisLayer;   // wa: result[0]?
            }

            return result;
        }
        /// <summary>
        /// Better in RandomProvider?
        /// </summary>
        float GetSmallRandomNumber()
        {
            return (float)(.0009 * rnd.NextDouble() + .0001) * (rnd.Next(2) == 0 ? -1 : 1);
        }

        #endregion

        #endregion

        #region properties

        public int[] NeuronsPerLayer { get; set; }
        public int L { get; }
        public Matrix[] w { get; set; }
        public Matrix[] b { get; set; }

        public float WeightsRange { get; set; }
        public float BiasRange { get; set; }

        #endregion
    }
}
