using System;
using System.Collections.Generic;
using System.IO;
using System.Net;

namespace FourPixCam
{
    /// <summary>
    /// https://stackoverflow.com/a/49407977
    /// </summary>
    public class MNISTReader
    {
        private const string 
            TrainImages = "http://yann.lecun.com/exdb/mnist/train-images.idx3-ubyte.gz",
            TrainLabels = "http://yann.lecun.com/exdb/mnist/train-labels.idx1-ubyte.gz",
            TestImages = "http://yann.lecun.com/exdb/mnist/t10k-images.idx3-ubyte.gz",
            TestLabels = "http://yann.lecun.com/exdb/mnist/t10k-labels.idx1-ubyte.gz";
        
        public static IEnumerable<MNISTImage> ReadTrainingData(string path = default)
        {
            foreach (var item in ReadFromUri(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }
        public static IEnumerable<MNISTImage> ReadTestData(string path = default)
        {
            // throw new ArgumentException("asd");
            foreach (var item in ReadFromUri(TestImages, TestLabels))
            {
                yield return item;
            }
        }
        public static IEnumerable<MNISTImage> ReadTrainingData(Stream stream)
        {
            foreach (var item in ReadFromStream(stream))
            {
                yield return item;
            }
        }
        public static IEnumerable<MNISTImage> ReadTestData(Stream stream)
        {
            // throw new ArgumentException("asd");
            foreach (var item in ReadFromStream(stream))
            {
                yield return item;
            }
        }

        static IEnumerable<MNISTImage> ReadFromUri(string imagesUri, string labelsUri)
        {
            byte[] imgs = new WebClient().DownloadData(imagesUri);
            using (var ms = new MemoryStream(imgs))
            {
                BinaryReader labels = new BinaryReader(new FileStream(labelsUri, FileMode.Open));
                BinaryReader images = new BinaryReader(new FileStream(imagesUri, FileMode.Open));

                int magicNumber = images.ReadBigInt32();
                int numberOfImages = images.ReadBigInt32();
                int width = images.ReadBigInt32();
                int height = images.ReadBigInt32();

                int magicLabel = labels.ReadBigInt32();
                int numberOfLabels = labels.ReadBigInt32();

                for (int i = 0; i < numberOfImages; i++)
                {
                    var bytes = images.ReadBytes(width * height);
                    var arr = new byte[height, width];

                    arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                    yield return new MNISTImage()
                    {
                        Data = arr,
                        Label = labels.ReadByte()
                    };
                }
            };
        }
        static IEnumerable<MNISTImage> ReadFromStream(Stream stream)
        {
            BinaryReader labels = new BinaryReader(stream);
            BinaryReader images = new BinaryReader(stream);

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new MNISTImage()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }
        }
    }
}
