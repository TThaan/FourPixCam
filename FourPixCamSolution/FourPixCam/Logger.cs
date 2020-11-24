using LINQPad;
using MatrixHelper;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace FourPixCam
{
    public static class Logger
    {
        #region fields

        static StreamWriter toFile = GetCustomWriter(@"c:\temp\FourPixTest.txt");
        static TextWriter toConsole = Console.Out;
        static TextWriter currentDisplay;

        #endregion

        #region public

        public enum Display
        {
            Standard,
            ToConsole, 
            ToFile,
            ToConsoleAndFile
        }
        public static Display StandardDisplay { get; set; } = Display.ToConsole;
        public static bool IsLogOn { get; set; }
        public static void SetError(Display display)
        {
            switch (display)
            {
                case Display.ToConsole:
                    Console.SetError(toConsole);
                    break;
                case Display.ToFile:
                    Console.SetError(toFile);
                    break;
            }
        }
        public static T Log<T>(this T obj, Display display = default)
        {
            return obj.Log("", display);
        }
        public static T Log<T>(this T obj, string prefix, Display display = default)
        {
            // Check if Logging is activated.
            if (!IsLogOn)
                return obj;

            if (display == Display.ToConsoleAndFile)
            {
                display = Display.ToConsole;
                SetDisplay(display);
                WriteToDisplay(obj, prefix);
                display = Display.ToFile;
            }

            SetDisplay(display);
            WriteToDisplay(obj, prefix);
            ResetDisplay();

            return obj;
        }
        public static void Log(float number, string prefix, Display display = default)
        {
            number.Log(prefix, display);
        }
        public static void Log(string text, Display display = default)
        {
            text.Log("", display);
        }
        /// <summary>
        /// 'objects' = pairs of value (string) name & value itself (object).
        /// </summary>
        public static void LogTitle(string text, char lineSymbol, Display display = default)
        {
            string titleText = string.Format(
                "\n\n    {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   \n" +
                $"                                         {text}" +
                "\n    {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   {0}   \n\n", lineSymbol);
            titleText.Log("", display);
        }
        /// <summary>
        /// 'objects' = pairs of value (string) name & value itself (object).
        /// </summary>

        #endregion

        #region helpers

        static StreamWriter GetCustomWriter(string path)
        {
            FileStream filestream = new FileStream(path, FileMode.Create);
            StreamWriter customWriter = new StreamWriter(filestream);
            customWriter.AutoFlush = true;

            return customWriter;
        }
        static void SetDisplay(Display display)
        {
            // Remember initial display.
            currentDisplay = Console.Out;

            // Set display to standard display if no other was passed as parameter.
            if (display == default)
            {
                display = StandardDisplay;
            }

            // Switch to desired display.
            switch (display)
            {
                case Display.ToConsole:
                    Console.SetOut(toConsole);
                    break;
                case Display.ToFile:
                    Console.SetOut(toFile);
                    break;
            }
        }
        static void ResetDisplay()
        {
            // Reset to initial display.
            Console.SetOut(currentDisplay);
        }
        static void WriteToDisplay<T>(T obj, string prefix)
        {
            // Write to desired display.
            if (typeof(T) == typeof(NeuralNet))
            {
                LogNeuralNet(obj as NeuralNet, prefix);
            }
            else if (typeof(T) == typeof(Sample[]))
            {
                LogSampleArray(obj as Sample[], prefix);
            }
            else if (typeof(T) == typeof(Sample))
            {
                LogSample(obj as Sample, prefix);
            }
            else if (typeof(T) == typeof(Matrix))
            {
                (obj as Matrix).DumpToConsole(prefix);
            }
            else if (typeof(T) == typeof(float))
            {
                Console.Write(prefix);
                Console.WriteLine(Convert.ToSingle(obj));
            }
            else if (typeof(T) == typeof(string))
            {
                Console.Write(prefix);
                Console.WriteLine(obj as string);
            }
            else if (typeof(T) == typeof(DateTime))
            {
                Console.Write(prefix);
                Console.WriteLine((DateTime)(object)obj);
            }
            else if (typeof(T) == typeof(TimeSpan))
            {
                Console.Write(prefix);
                Console.WriteLine((TimeSpan)(object)obj);
            }
            else
            {
                throw new NotImplementedException();
            }
        }
        static NeuralNet LogNeuralNet(NeuralNet net, string prefix = "")
        {
            Log(prefix);

            Log($"\n                                    T H E   N E U R A L   N E T");
            Log($"                                  - - - - - - - - - - - - - - - -\n");
            Log($"                                    NeuronsPerLayer : {net.Layers.Select(x=>x.N).ToCollectionString()}");

            for (int i = 0; i < net.LayersCount; i++)
            {
                Console.WriteLine($"    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   \n");
                Log($"\n                                            L a y e r  {i}");
                Log($"                                  - - - - - - - - - - - - - - - -\n");
                Log($"                                    Neurons   : {net.Layers[i].N}");
                Log($"                                    Activator : {net.Layers[i].ActivationType}");

                Matrix w = net.Layers[i].Weights;
                if (w != null)
                {
                    w.Log($"\nw = ");
                }

                if (net.Layers[i].Biases != null)
                {
                    Matrix b = net.Layers[i].Biases;
                    b.Log($"\nb = ");
                }
            }

            return net;
        }
        static Sample[] LogSampleArray(Sample[] samples, string prefix = "")
        {
            Log(prefix);

            foreach (var sample in samples)
            {
                sample.Log();
            }

            return samples;
        }
        static Sample LogSample(Sample sample, string prefix = "")
        {
            Log(prefix);

            Log("\n +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n");
            sample.RawInput.Log($"\n{nameof(sample.RawInput)} = ");
            sample.Input.Log($"\n{nameof(sample.Input)} = ");
            sample.ActualOutput.Log($"\n{nameof(sample.ActualOutput)} = ");
            sample.ExpectedOutput.Log($"\n{nameof(sample.ExpectedOutput)} = ");
            Log($"{nameof(sample.Label)} = {sample.Label}");
            if (sample.IsOutputCorrect == true)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            Log($"\n{nameof(sample.IsOutputCorrect)} = {sample.IsOutputCorrect}\n");
            Console.ResetColor();
            // Console.WriteLine(" +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +   -   +\n\n");
            return sample;
        }

        #endregion

        #region unused?

        // Dumps to debugger. Used in the HTML debug view.
        public static T DumpToHTMLDebugger<T>(this T o, out string result, string title = "")
        {
            TextWriter writer = Util.CreateXhtmlWriter();
            writer.Write(o);
            result = writer.ToString();
            return o;
        }
        // Dumps to a temporary html file and opens in the browser (LinqPad).
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
            Process.Start(new ProcessStartInfo(localUrl) { UseShellExecute = true });
        }

        #endregion
    }
}
