using System;
using System.Collections.Generic;
using System.Linq;

namespace MLtestapp
{
    public class Model
    {
        public int Id;
        public double GeneOneScore;
        public double GeneTwoScore;
    }

    public class FeatureModel
    {
        public float GeneOneScore;
        public float GeneTwoScore;
    }

    public static class Extensions
    {
        public static IEnumerable<FeatureModel> Project(this IEnumerable<Model> model)
        {
            return model.Select(o => new FeatureModel { GeneOneScore = (float)o.GeneOneScore, GeneTwoScore = (float)o.GeneTwoScore });
        }
    }

    class DataGenerator
    {
        public readonly int _totalSamples = 200;

        public readonly double _scoreOneMin = 0.8;

        public readonly double _scoreTwoMin = 0.1;

        public readonly double _scoreOneVariance = 0.05;
        public readonly double _scoreTwoVariance = 0.4;

        private readonly Random p1;
        private readonly Random p2;

        public DataGenerator()
        {
            p1 = new Random(980665);
            p2 = new Random(314159);
        }

        public List<Model> Generate()
        {
            var result = new List<Model>();

            for (int i = 0; i < _totalSamples; i++)
            {
                var sample = new Model
                {
                    Id = i,
                    GeneOneScore = GenerateGeneScore(p1, _scoreOneMin, _scoreOneVariance),
                    GeneTwoScore = GenerateGeneScore(p2, _scoreTwoMin, _scoreTwoVariance),
                };

                result.Add(sample);
            }

            return result;
        }

        public double GenerateGeneScore(Random generator, double min, double range)
        {
            return min + range * ((double)generator.Next(1000) / 1000.0);
        }
    }

    public class ResultDisplay
    {
        public float GeneOneScore;
        public float GeneTwoScore;
        public float[] NormalisedFeatures { get; set; }
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
    }
}
