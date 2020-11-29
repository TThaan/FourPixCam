using MatrixHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FourPixCam
{
    internal class DataFactory
    {
        #region ctor & fields

        #endregion

        #region public

        public static Sample[] GetData(Stream stream)
        {
            Sample[] result;
            byte[] tmpResult;

            using (MemoryStream ms = new MemoryStream())
            {
                stream.CopyTo(ms);
                tmpResult = ms.ToArray();
            }

            var header = tmpResult.Take(16).Reverse().ToArray();
            int imgCount = BitConverter.ToInt32(header, 8);
            int rows = BitConverter.ToInt32(header, 4);
            int cols = BitConverter.ToInt32(header, 0);

            return null;//result
        }

        #endregion
    }
}
