using LINQPad;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace FourPixCam
{
    public static class ExtensionMethods
    {
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
        {/*
            for (int i = 0; i < net.Layers.Count(); i++)
            {
                Layer layer = net.ElementAt(i);

                Console.WriteLine($"Layer ID     : {i}");
                Console.WriteLine($"Count        : {layer.Count()}");
                Console.WriteLine($"WeightsRange : {layer.WeightsRange}");
                Console.WriteLine($"BiassRange   : {layer.BiasRange}");
                Console.WriteLine($"Activator    : {(layer.Activator != null ? layer.Activator.ToString() : "null")}\n");

                string weightsString = "";
                string biasesString = "";

                if (layer.w != null)
                {
                    for (int weight = 0; weight < layer.w.Length; weight++)
                    {
                        // as extension method:
                        foreach (double value in layer.w)
                            {
                                weightsString += value + ", ";
                            }
                            weightsString.Substring(0, weightsString.Length - 2);
                    }

                    for (int bias = 0; bias < layer.b.Length; bias++)
                    {
                        // as extension method:
                        foreach (double value in layer.b)
                        {
                            biasesString += value + ", ";
                        }
                        biasesString.Substring(0, biasesString.Length - 2);
                    }
                    Console.WriteLine($"               Bias         : {biasesString}");
                    Console.WriteLine($"               Weights      : {weightsString}");

                    Console.WriteLine("\n");
                }
            }

            if (waitForEnter)
            {
                Console.ReadLine();
            }*/
            return net;
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
