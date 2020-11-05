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

        static DataFactory()
        {
            rnd = RandomProvider.GetThreadRandom();
        }

        #endregion

        internal static Sample[] GetTrainingData(int sampleSize)
        {
            return Enumerable.Range(0, sampleSize)
                .Select(x => GetRandomSample())
                .ToArray();
        }
        //internal static Sample[] GetTestingData()
        //{


        //}

        #region helper methods

        static Sample GetRandomSample()
        {
            var inp = GetInput();
            var exp = GetExpectedOutput(inp);

            return new Sample
            {
                Input = inp,
                ExpectedOutput = exp
            };
        }
        static Matrix GetInput()
        {
            return new Matrix(new[]
                    {
                        (float)Math.Round(rnd.NextDouble()),
                        (float)Math.Round(rnd.NextDouble()),
                        (float)Math.Round(rnd.NextDouble()),
                        (float)Math.Round(rnd.NextDouble())
                    });
        }
        static Matrix GetExpectedOutput(Matrix input)
        {
            return input;
        }
        //static Dictionary<float[], string> GetExpectedResult()
        //{
        //    Dictionary<float[], string> result = new Dictionary<float[], string>();

        //    foreach (float[] trainingSample in trainingData)
        //    {
        //        result[trainingSample] = GetLabel(trainingSample);
        //    }

        //    return result;
        //}
        static string GetLabel(float[] sample)
        {
            float _0 = sample[0];
            float _1 = sample[1];
            float _2 = sample[2];
            float _3 = sample[3];

            if (_0 == _1)
            {
                if (_2 == _3)
                {
                    if (_0 == _2)
                    {
                        return _0 == 0
                            ? "AllBlack"
                            : "AllWhite";
                    }
                    else
                    {
                        return _0 == 0
                            ? "Black Top - White Bottom (hori)"
                            : "White Top - Black Bottom (hori)";
                    }
                }
            }
            else if (_0 == _2)
            {
                if (_1 == _3)
                {
                    return _0 == 0
                        ? "Black Left - White Right (vert)"
                        : "White Left - Black Right (vert)";
                }
            }
            else if (_0 == _3)
            {
                if (_1 == _2)
                {
                    return _0 == 0
                        ? "Black TopLeft & RightBottom (diag)"
                        : "White TopLeft & RightBottom (diag)";
                }
            }

            return "No match.";
        }

        #endregion
    }
}
