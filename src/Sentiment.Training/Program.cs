using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace Sentiment.Training
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load data

            int min_count = 10;
            double polarity_cutoff = 0.1;

            var labels = File.ReadAllLines("../../../../Data/labels.txt");
            var reviews = File.ReadAllLines("../../../../Data/reviews.txt");

            // Count words

            var vocab = new Dictionary<string, int>();
            var positive_counts = new Dictionary<string, int>();
            var negative_counts = new Dictionary<string, int>();
            var pos_neg_ratios = new Dictionary<string, double>();

            foreach (var pair in reviews.Zip(labels, (review, label) => new { review, label }))
            {
                var review = pair.review;
                var label = pair.label;

                foreach (var word in review.ToLower().Split(' '))
                {
                    vocab.TryGetValue(word, out var count);
                    vocab[word] = count + 1;

                    var dico = label == "positive" ? positive_counts : negative_counts;
                    dico.TryGetValue(word, out count);
                    dico[word] = count + 1;

                    var otherDico = label == "positive" ? negative_counts : positive_counts;
                    otherDico.TryGetValue(word, out count);
                    otherDico[word] = count; // This is used to set count to 0 words that appear only on one side
                }
            }

            // Compute ratios

            foreach (var word in vocab.Keys)
            {
                if (vocab[word] > 50)
                {
                    var ratio = positive_counts[word] / (negative_counts[word] + 1.0);
                    if (ratio > 1.0)
                    {
                        pos_neg_ratios[word] = Math.Log(ratio);
                    }
                    else
                    {
                        pos_neg_ratios[word] = -Math.Log(1.0 / (ratio + 0.01));
                    }
                }
                else
                {
                    pos_neg_ratios[word] = 0.0;
                }
            }

            var review_vocab = vocab.Where(o => o.Value > min_count && Math.Abs(pos_neg_ratios[o.Key]) > polarity_cutoff).Select(o => o.Key).ToList();

            // Create word to index map

            var wordToIndex = review_vocab.Select((word, index) => new { word, index }).ToDictionary(o => o.word, o => o.index);

            // Build network

            var network = new Net<double>();
            network.AddLayer(new InputLayer(1, 1, review_vocab.Count));
            network.AddLayer(new FullyConnLayer(10));
            network.AddLayer(new FullyConnLayer(1));
            network.AddLayer(new RegressionLayer());

            // Training

            var trainer = new SgdTrainer(network)
            {
                LearningRate = 0.005
            };

            var input = BuilderInstance.Volume.SameAs(new Shape(1, 1, review_vocab.Count));
            var output = BuilderInstance.Volume.SameAs(new Shape(1, 1, 1));

            int i = 0;
            int correct = 0;

            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine($"Epoch #{epoch}");

                foreach (var pair in reviews.Zip(labels, (review, label) => new { review, label }))
                {
                    var review = pair.review;
                    var label = pair.label;
                    FillVolume(input, review, wordToIndex);

                    output.Set(0, 0, 0, pair.label == "positive" ? 1.0 : 0.0);

                    var test = network.Forward(input);
                    if (test > 0.5 && label == "positive" || test < 0.5 && label == "negative")
                    {
                        correct++;
                    }

                    trainer.Train(input, output);

                    if (i % 100 == 0)
                    {
                        Console.WriteLine($"Accuracy: {Math.Round(correct / (double)i * 100.0, 2) }%");
                        Console.WriteLine($"{i}/{reviews.Length}");
                    }

                    i++;
                    if (Console.KeyAvailable) break;
                }
            }

            // Save Network

            File.WriteAllText(@"../../../../Model/sentiment.json", network.ToJson());
        }

        private static void FillVolume(Volume<double> input, string review, Dictionary<string, int> wordToIndex)
        {
            input.Clear();
            foreach (var word in review.Split(' '))
            {
                if (wordToIndex.TryGetValue(word, out var index))
                {
                    input.Set(0, 0, index, 1.0);
                }
            }
        }
    }
}
