using LINQPad;
using MatrixHelper;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace FourPixCam
{
    public static class ExtensionMethods
    {
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
        public static NeuralNet DumpToConsole(this NeuralNet net, bool waitForEnter = false)
        {
            Console.WriteLine($"\n                                    T H E   N E U R A L   N E T");
            Console.WriteLine($"                                  - - - - - - - - - - - - - - - -\n");
            Console.WriteLine($"                                    NeuronsPerLayer : {net.WeightRange}");
            Console.WriteLine($"                                    WeightRange     : {net.WeightRange}");
            Console.WriteLine($"                                    BiasRange       : {net.BiasRange}");
            Console.WriteLine();

            for (int i = 0; i < net.L; i++)
            {
                //Console.WriteLine($"    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   \n");
                Console.WriteLine($"\n                                            L a y e r  {i}");
                Console.WriteLine($"                                  - - - - - - - - - - - - - - - -\n");
                Console.WriteLine($"                                    Neurons   : {net.NeuronsPerLayer[i]}");
                Console.WriteLine($"                                    Activator : {net.ActivationTypes[i]}");

                Matrix w = net.W[i];
                if (w != null)
                {
                    w.DumpToConsole($"\nw = ");
                }

                Matrix b = net.B[i];
                if (b != null)
                {
                    b.DumpToConsole($"\nb = ");
                }
            }

            if (waitForEnter)
            {
                Console.ReadLine();
            }
            return net;
        }
        public static Sample[] DumpToConsole(this Sample[] samples, bool waitForEnter = false)
        {
            foreach (var sample in samples)
            {
                sample.DumpToConsole();
            }

            if (waitForEnter)
            {
                Console.ReadLine();
            }
            return samples;
        }
        public static Sample DumpToConsole(this Sample sample, bool waitForEnter = false)
        {

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
    }
}
