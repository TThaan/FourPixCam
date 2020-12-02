using System;
using System.Collections.Generic;
using System.Linq;

namespace FourPixCam
{
    internal static class ExtensionMethods
    {
        internal static string ToCollectionString<T>(this IEnumerable<T> collection)
        {
            return string.Join(",", collection.Select(x => x.ToString()));
        }
        internal static string ToVerticalCollectionString<T>(this IEnumerable<T> collection)
        {
            return string.Join("\n", collection.Select(x => x.ToString()));
        }
        internal static List<T> ToList<T>(this Array arr)
        {
            var result = new List<T>();

            for (int i = 0; i < arr.Length; i++)
            {
                result.Add((T)arr.GetValue(i));
            }

            return result;
        }
        internal static IEnumerable<T> Shuffle<T>(this IEnumerable<T> collection)
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
