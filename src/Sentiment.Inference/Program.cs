using System;

namespace Sentiment.Inference
{

    class Program
    {
        static void Main(string[] args)
        {
            var sentimentInference = new SentimentInference();

            // Get review from user

            Console.WriteLine("Review:" + Environment.NewLine);
            var review = Console.ReadLine();

            // Inference

            var result = sentimentInference.Infer(review);
            Console.WriteLine(result);
        }
    }
}
