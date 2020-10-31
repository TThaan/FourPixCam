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
            Layers = layers;
            Length = layers.Count();

            WeightsRange = 2;
            BiasRange = 5;

            w = GetWeights();
            b = GetBiases();
        }

        #region helper methods

        /// <summary>
        /// first dimension/ vertical interpretation = projective field/next layer
        /// </summary>
        /// <returns></returns>
        Matrix[] GetWeights()
        {
            Matrix[] result = new Matrix[Length];

            // iterate over layers (skip layer[1])
            for (int i = 1; i < Length; i++)
            {
                Matrix weightMatrix = new Matrix(Layers[i], Layers[i - 1]);

                // iterate over neurons per this layer
                // weight matrix = only matrix where "this" neurons are horizontally aligned ?
                for (int n = 0; n < Layers[i]; n++)
                {
                    // iterate over neurons per previous/receptive layer
                    for (int m = 0; m < Layers[i-1]; m++)
                    {
                        weightMatrix[n, m] = WeightsRange / 2 * GetSmallRandomNumber();
                    }
                };

                result[i] = weightMatrix;   // wa: result[0]?
            }

            return result;
        }
        /// <summary>
        /// Why are all first neurons biases 0?
        /// </summary>
        /// <returns></returns>
        Matrix[] GetBiases()
        {
            Matrix[] result = new Matrix[Length];

            // iterate over layers (skip layer[1])
            for (int i = 1; i < Length; i++)
            {
                Matrix biasMatrix = new Matrix(Layers[i], 1);

                // iterate over neurons per this layer
                for (int m = 0; m < Layers[i]; m++)
                {
                    biasMatrix[m, 0] = BiasRange / 2 * GetSmallRandomNumber();
                };

                result[i] = biasMatrix;   // wa: result[0]?
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

        public int[] Layers { get; set; }
        public int Length { get; }
        public Matrix[] w { get; set; }
        public Matrix[] b { get; set; }

        public float WeightsRange { get; set; }
        public float BiasRange { get; set; }

        #endregion
    }
}
