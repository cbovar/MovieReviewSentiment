using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace Sentiment.Inference
{
    public class SentimentInference
    {
        private Dictionary<string, int> wordToIndex;
        private Net<double> network;
        private List<string> review_vocab;

        public SentimentInference()
        {
            // Load data

            int min_count = 10;
            double polarity_cutoff = 0.1;

            var labels = File.ReadAllLines("../Data/labels.txt");
            var reviews = File.ReadAllLines("../Data/reviews.txt");

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

            this.review_vocab = vocab.Where(o => o.Value > min_count && Math.Abs(pos_neg_ratios[o.Key]) > polarity_cutoff).Select(o => o.Key).ToList();

            // Create word to index map

            this.wordToIndex = review_vocab.Select((word, index) => new { word, index }).ToDictionary(o => o.word, o => o.index);

            // Load network

            this.network = ConvNetSharp.Core.Serialization.SerializationExtensions.FromJson<double>(File.ReadAllText("../Model/sentiment.json"));
        }

        public double Infer(string review)
        {
            var input = BuilderInstance.Volume.SameAs(new Shape(1, 1, this.review_vocab.Count));

            FillVolume(input, review);

            return network.Forward(input);
        }

        private void FillVolume(Volume<double> input, string review)
        {
            input.Clear();
            foreach (var word in review.Split(' '))
            {
                if (this.wordToIndex.TryGetValue(word, out var index))
                {
                    input.Set(0, 0, index, 1.0);
                }
            }
        }
    }
}
