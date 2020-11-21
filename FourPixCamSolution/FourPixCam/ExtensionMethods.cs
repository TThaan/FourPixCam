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

        // Into ILogger?:

        public static T LogIt<T>(this T obj, string title = "")//, bool isLogged = false
        {
            // Into Logger !
            if (!NeurNetMath.IsLogOn)
                return obj;

            if (typeof(T) == typeof(NeuralNet))
            {
                (obj as NeuralNet).DumpToConsole();
            }
            else if (typeof(T) == typeof(Sample[]))
            {
                (obj as Sample[]).DumpToConsole();
            }
            else if (typeof(T) == typeof(Sample))
            {
                (obj as Sample).DumpToConsole();
            }
            else if (typeof(T) == typeof(Matrix))
            {
                (obj as Matrix).DumpToConsole(title);
            }
            else if (typeof(T) == typeof(float))
            {
                // obj as object ? (vgl: 'WriteDumpingTitle')
                // throw new NotImplementedException();
                Convert.ToSingle(obj).DumpToConsole(title);
            }
            else
            {
                // obj as object ? (vgl: 'WriteDumpingTitle')
                throw new NotImplementedException();
                // (obj as object).DumpToConsole(title);
            }

            return obj;
        }

        // Dumps to a temporary html file and opens in the browser.
        public static void DumpToExplorer<T>(this T o, string title = "")
        {
            object obj;
            if (string.IsNullOrWhiteSpace(title))
            {
                obj = o;
            }
            else
            {

                obj = new { Comment = title, Object = o };
            }

            string localUrl = Path.GetTempFileName() + ".html";
            using (TextWriter writer = Util.CreateXhtmlWriter(true))
            {
                writer.Write(obj);
                string s = writer.ToString();
                File.WriteAllText(localUrl, s);
            }
            Process.Start(new ProcessStartInfo(localUrl) { UseShellExecute = true} );
        }
        public static NeuralNet DumpToConsole(this NeuralNet net)
        {
            Console.WriteLine($"\n                                    T H E   N E U R A L   N E T");
            Console.WriteLine($"                                  - - - - - - - - - - - - - - - -\n");
            Console.WriteLine($"                                    NeuronsPerLayer : {net.NeuronsPerLayer.ToCollectionString()}");
            Console.WriteLine($"                                    IsWithBias      : {net.IsWithBias}");
            Console.WriteLine();

            for (int i = 0; i < net.LayerCount; i++)
            {
                //Console.WriteLine($"    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   \n");
                Console.WriteLine($"\n                                            L a y e r  {i}");
                Console.WriteLine($"                                  - - - - - - - - - - - - - - - -\n");
                Console.WriteLine($"                                    Neurons   : {net.NeuronsPerLayer[i]}");
                Console.WriteLine($"                                    Activator : {net.Activations[i]?.Method.DeclaringType.Name}");

                Matrix w = net.W[i];
                if (w != null)
                {
                    w.DumpToConsole($"\nw = ");
                }

                if (net.IsWithBias)
                {
                    Matrix b = net.B[i];
                    b.DumpToConsole($"\nb = ");
                }
            }

            return net;
        }
        public static Sample[] DumpToConsole(this Sample[] samples)
        {
            foreach (var sample in samples)
            {
                sample.DumpToConsole();
            }

            return samples;
        }
        public static Sample DumpToConsole(this Sample sample)
        {
            Console.WriteLine("\n +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n");
            sample.RawInput.DumpToConsole($"\n{nameof(sample.RawInput)} = ");
            sample.Input.DumpToConsole($"\n{nameof(sample.Input)} = ");
            sample.ActualOutput.DumpToConsole($"\n{nameof(sample.ActualOutput)} = ");
            sample.ExpectedOutput.DumpToConsole($"\n{nameof(sample.ExpectedOutput)} = ");
            Console.WriteLine($"{nameof(sample.Label)} = {sample.Label}");
            if (sample.IsOutputCorrect == true)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            Console.WriteLine($"\n{nameof(sample.IsOutputCorrect)} = {sample.IsOutputCorrect}\n");
            Console.ResetColor();
            // Console.WriteLine(" +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n\n");
            return sample;
        }
        public static object DumpToConsole(this float number, string title = "")
        {
            Console.WriteLine($"\n{title} = {number.ToString()}\n");
            return number;
        }
        //public static object DumpToConsole(this object obj, string title = "")
        //{
        //    Console.WriteLine($"\n{title} = {obj.ToString()}\n");            
        //    return obj;
        //}
        // Dumps to debugger. Used in the HTML debug view.
        public static T DumpToHTMLDebugger<T>(this T o, out string result, string title = "")
        {
            TextWriter writer = Util.CreateXhtmlWriter();
            writer.Write(o);
            result = writer.ToString();
            return o;
        }


        /// <summary>
        /// 'objects' = pairs of value (string) name & value itself (object).
        /// </summary>
        public static void WriteDumpingTitle(this string title, params object[] objects)
        {
            Console.WriteLine("\n\n    *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine($"                                         {title}");
            Console.WriteLine("\n    *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n\n");

            for (int i = 0; i < objects.Length; i+=2)
            {
                string s;
                try
                {
                    s = (string)objects[i+1];
                }
                catch (Exception)
                {
                    s = objects[i+1].ToString();
                }
                Console.WriteLine($"                                    {objects[i] as string}   : {s}");
            }
        }
        /// <summary>
        /// 'objects' = pairs of value (string) name & value itself (object).
        /// </summary>
        public static void WriteDumpingTitle(this string title, string commentInBrackets)
        {
            Console.WriteLine("\n\n    *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine($"                                         {title}   ({commentInBrackets})");
            Console.WriteLine("\n    *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n\n");
        }
    }
}
