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
        public static NeuralNet DumpToConsole(this NeuralNet net, bool dumpWhileWorking)
        {
            if (!dumpWhileWorking)
                return net;

            Console.WriteLine($"\n                                    T H E   N E U R A L   N E T");
            Console.WriteLine($"                                  - - - - - - - - - - - - - - - -\n");
            Console.WriteLine($"                                    NeuronsPerLayer : {net.NeuronsPerLayer.ToCollectionString()}");
            Console.WriteLine($"                                    WeightMin     : {net.WeightMin}");
            Console.WriteLine($"                                    WeightMax     : {net.WeightMax}");
            Console.WriteLine($"                                    BiasMin       : {net.BiasMin}");
            Console.WriteLine($"                                    BiasMax       : {net.BiasMax}");
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
                    w.DumpToConsole($"\nw = ", dumpWhileWorking);
                }

                if (net.IsWithBias)
                {
                    Matrix b = net.B[i];
                    b.DumpToConsole($"\nb = ", dumpWhileWorking);
                }
            }

            return net;
        }
        public static Sample[] DumpToConsole(this Sample[] samples, bool dumpWhileWorking)
        {
            if (!dumpWhileWorking)
                return samples;

            foreach (var sample in samples)
            {
                sample.DumpToConsole(null, false, dumpWhileWorking);
            }

            return samples;
        }
        public static Sample DumpToConsole(this Sample sample, Matrix actualOutput, bool isCorrect, bool dumpWhileWorking)
        {
            if (!dumpWhileWorking)
                return sample;

            Console.WriteLine("\n +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n");
            sample.RawInput.DumpToConsole($"\n{nameof(sample.RawInput)} = ", dumpWhileWorking);
            sample.Input.DumpToConsole($"\n{nameof(sample.Input)} = ", dumpWhileWorking);
            actualOutput.DumpToConsole($"\n{nameof(actualOutput)} = ", dumpWhileWorking);
            sample.ExpectedOutput.DumpToConsole($"\n{nameof(sample.ExpectedOutput)} = ", dumpWhileWorking);
            Console.WriteLine($"{nameof(sample.Label)} = {sample.Label}", dumpWhileWorking);
            if (isCorrect)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            Console.WriteLine($"\n{nameof(isCorrect)} = {isCorrect}\n");
            Console.ResetColor();
            // Console.WriteLine(" +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n\n");
            return sample;
        }
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
