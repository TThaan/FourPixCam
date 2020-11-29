using LINQPad;
using MatrixHelper;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace FourPixCam
{
    public static class ExtensionMethods
    {
        public static string ToCollectionString<T>(this IEnumerable<T> collection)
        {
            return string.Join(",", collection.Select(x => x.ToString()));
        }
        public static string ToVerticalCollectionString<T>(this IEnumerable<T> collection)
        {
            return string.Join("\n", collection.Select(x => x.ToString()));
        }
        public static List<T> ToList<T>(this Array arr)
        {
            var result = new List<T>();

            for (int i = 0; i < arr.Length; i++)
            {
                result.Add((T)arr.GetValue(i));
            }

            return result;
        }
        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> collection)
        {
            Random rnd = new Random();

            T[] result = collection.ToArray();
            int count = collection.Count();

            for (int index = 0; index < count; index++)
            {
                int newIndex = rnd.Next(count);

                // Exchange arr[n] with arr[k]

                T item = result[index];
                result[index] = result[newIndex];
                result[newIndex] = item;
            }

            return result;
        }
    }
}
