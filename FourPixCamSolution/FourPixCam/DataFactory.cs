using MatrixHelper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace FourPixCam
{
    internal class DataFactory
    {
        #region ctor & fields

        static Random rnd;
        static Dictionary<Label, Matrix> rawInputs;
        static Dictionary<Label, Matrix> noisyInputs;
        static Dictionary<Label, Matrix> validInputs;
        static Dictionary<Label, Matrix> validOutputs;
        static Sample[] validSamples;

        #endregion

        internal static Sample[] GetTrainingData(int sampleSize)
        {
            rnd = RandomProvider.GetThreadRandom();

            rawInputs = GetRawInputs();
            noisyInputs = GetNoisyInputs();
            validInputs = GetValidInputs(rawInputs);//noisyInputs
            validOutputs = GetValidOutputs();
            validSamples = GetValidSamples();

            return GetValidTrainingData(sampleSize, validSamples);
        }

        private static Sample[] GetValidTrainingData(int sampleSize, Sample[] _validSamples)
        {
            List<Sample> tmpResult = new List<Sample>();
            int amountOfCompleteSampleSets = (int)Math.Round((double)sampleSize / rawInputs.Values.Count, 0);

            for (int i = 0; i < amountOfCompleteSampleSets; i++)
            {
                tmpResult.AddRange(_validSamples);
            }
            Sample[] result = tmpResult.ToArray();
            Shuffle(result);

            // debug
            //var debug = result.GroupBy(x => x.Label);
            //if (debug.Any(y=>y.Count() < amountOfCompleteSampleSets-1))
            //{

            //}
            //List<int> counts = new List<int>();
            //foreach (var item in debug)
            //{
            //    counts.Add(item.Count());
            //}
            return result;
        }
        static void Shuffle(Sample[] trainingData)
        {
            int n = trainingData.Length;

            while (n > 1)
            {
                int k = rnd.Next(n--);

                // Exchange arr[n] with arr[k]

                Sample temp = trainingData.ElementAt(n);
                trainingData[n] = trainingData[k];
                trainingData[k] = temp;
            }
        }

        internal static Sample[] GetTestingData(int multiplyer)
        {
            var result = new List<Sample>();
            for (int i = 0; i < multiplyer; i++)
            {
                result.AddRange(validSamples);
            }
            return result.ToArray();
        }

        #region helper methods

        static Sample[] GetValidSamples()
        {
            var result = new List<Sample>();

            // int vs Label ? .Select(x => (Label)x)?
            var labels = Enum.GetValues(typeof(Label)).ToList<Label>().Skip(1); 
            foreach (var label in labels)
            {
                result.Add(new Sample
                {
                    Label = label,
                    RawInput = rawInputs[label],
                    Input = validInputs[label],
                    ExpectedOutput = validOutputs[label]
                });
            }

            return result.ToArray();
        }
        static Dictionary<Label, Matrix> GetRawInputs()
        {
            return new Dictionary<Label, Matrix>
            {
                [Label.AllBlack] = new Matrix(new float[,] {
                    { -1, -1 },
                    { -1, -1 } }),

                [Label.AllWhite] = new Matrix(new float[,] {
                    { 1, 1 },
                    { 1, 1 } }),

                [Label.TopBlack] = new Matrix(new float[,] {
                    { -1, -1 },
                    { 1, 1 } }),

                [Label.TopWhite] = new Matrix(new float[,] {
                    { 1, 1 },
                    { -1, -1 } }),

                [Label.LeftBlack] = new Matrix(new float[,] {
                    { -1, 1 },
                    { -1, 1 } }),

                [Label.LeftWhite] = new Matrix(new float[,] {
                    { 1, -1 },
                    { 1, -1 } }),

                [Label.SlashBlack] = new Matrix(new float[,] {
                    { 1, -1 },
                    { -1, 1 } }),

                [Label.SlashWhite] = new Matrix(new float[,] {
                    { -1, 1 },
                    { 1, -1 } })
            };
        }
        static Dictionary<Label, Matrix> GetNoisyInputs()
        {
            return new Dictionary<Label, Matrix>
            {
                [Label.AllBlack] = new Matrix(new float[,] {
                    { -(GetNoisyValue()), -(GetNoisyValue()) },
                    { -(GetNoisyValue()), -(GetNoisyValue()) } }),

                [Label.AllWhite] = new Matrix(new float[,] {
                    { (GetNoisyValue()), (GetNoisyValue()) },
                    { (GetNoisyValue()), (GetNoisyValue()) } }),

                [Label.TopBlack] = new Matrix(new float[,] {
                    { -(GetNoisyValue()), -(GetNoisyValue()) },
                    { (GetNoisyValue()), (GetNoisyValue()) } }),

                [Label.TopWhite] = new Matrix(new float[,] {
                    { (GetNoisyValue()), (GetNoisyValue()) },
                    { -(GetNoisyValue()), -(GetNoisyValue()) } }),

                [Label.LeftBlack] = new Matrix(new float[,] {
                    { -(GetNoisyValue()), (GetNoisyValue()) },
                    { -(GetNoisyValue()), (GetNoisyValue()) } }),

                [Label.LeftWhite] = new Matrix(new float[,] {
                    { (GetNoisyValue()), -(GetNoisyValue()) },
                    { (GetNoisyValue()), -(GetNoisyValue()) } }),

                [Label.SlashBlack] = new Matrix(new float[,] {
                    { (GetNoisyValue()), -(GetNoisyValue()) },
                    { -(GetNoisyValue()), (GetNoisyValue()) } }),

                [Label.SlashWhite] = new Matrix(new float[,] {
                    { -(GetNoisyValue()), (GetNoisyValue()) },
                    { (GetNoisyValue()), -(GetNoisyValue()) } })
            };
        }
        static float GetNoisyValue()
        {
            return 1f - (float)rnd.NextDouble() / 3f;
        }

        static Dictionary<Label, Matrix> GetValidInputs(Dictionary<Label, Matrix> _rawInputs)
        {
            var test = _rawInputs.ToDictionary(x => x.Key, x => Operations.FlattenToOneColumn(x.Value));
            return test;
        }
        static Dictionary<Label, Matrix> GetValidOutputs()
        {
            return new Dictionary<Label, Matrix>
            {
                [Label.AllWhite] = new Matrix(new float[] { 1, 0, 0, 0 }),

                [Label.AllBlack] = new Matrix(new float[] { 1, 0, 0, 0 }),

                [Label.TopWhite] = new Matrix(new float[] { 0, 1, 0, 0 }),

                [Label.TopBlack] = new Matrix(new float[] { 0, 1, 0, 0 }),

                [Label.LeftWhite] = new Matrix(new float[] { 0, 0, 1, 0 }),

                [Label.LeftBlack] = new Matrix(new float[] { 0, 0, 1, 0 }),

                [Label.SlashWhite] = new Matrix(new float[] { 0, 0, 0, 1 }),

                [Label.SlashBlack] = new Matrix(new float[] { 0, 0, 0, 1 })
            };
        }
        static Sample GetRandomValidSample()
        {
            // int AllBlack, AllWhite, LeftBlack
            var debug = validSamples.ElementAt(rnd.Next(0, validInputs.Count()));
            switch (debug.Label)
            {
                case Label.Undefined:
                    break;
                case Label.AllBlack:
                    break;
                case Label.AllWhite:
                    break;
                case Label.LeftBlack:
                    break;
                case Label.LeftWhite:
                    break;
                case Label.SlashBlack:
                    break;
                case Label.SlashWhite:
                    break;
                case Label.TopBlack:
                    break;
                case Label.TopWhite:
                    break;
                default:
                    break;
            }
            return debug;
        }

        #endregion
    }
}
