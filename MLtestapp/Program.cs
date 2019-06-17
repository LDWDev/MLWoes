using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace MLtestapp
{
    class Program
    {
        static void Main(string[] args)
        {
            var generator = new DataGenerator();

            var rawData = generator.Generate();

            RunRandomisedPca(rawData);

            Console.ReadLine();
        }

        static void RunRandomisedPca(List<Model> data)
        {
            MLContext ct = new MLContext();
            var dataProjection = data.Project().Take(50);

            IDataView trainingData = ct.Data.LoadFromEnumerable<FeatureModel>(dataProjection);

            var rpcaProjection = ct.Transforms.Concatenate("Features", "GeneOneScore", "GeneTwoScore")
                  .Append(ct.Transforms.NormalizeMeanVariance("NormalisedFeatures", "Features"))
                .Append(ct.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: "NormalisedFeatures", rank: 2));

            var fitOfTrainingData = rpcaProjection.Fit(trainingData);

            var transformOfTrainingData = fitOfTrainingData.Transform(trainingData);

            // gene one is constrained to be within .8 and .9, and gene two between .1 and .5
            var anomaly = new List<FeatureModel> { new FeatureModel { GeneOneScore = (float)100000, GeneTwoScore = (float)25000 } };

            var iDataViewOfAnomaly = ct.Data.LoadFromEnumerable<FeatureModel>(anomaly);
            var transformOfAnomaly = fitOfTrainingData.Transform(iDataViewOfAnomaly);

            var trainingDataResults = ct.Data.CreateEnumerable<ResultDisplay>(transformOfTrainingData, reuseRowObject: false).ToList();
            var anomalousDataResult = ct.Data.CreateEnumerable<ResultDisplay>(transformOfAnomaly, reuseRowObject: false).ToList();

            Console.WriteLine("Results from transforming first 20 training data: Predicted, score, PCA co-ordinates");
            foreach (var r in trainingDataResults.Take(20))
            {
                Console.WriteLine(r.PredictedLabel + ", " + r.Score + ", " + r.NormalisedFeatures[0] + ", " + r.NormalisedFeatures[1]);
            }

            // Claims this has a similar score to the other data, despite completely different co-ordinates.
            Console.WriteLine("Results from transforming the \"anomaly\": Predicted, score, PCA co-ordinates");
            foreach (var r in anomalousDataResult)
            {
                Console.WriteLine(r.PredictedLabel + ", " + r.Score + ", " + r.NormalisedFeatures[0] + ", " + r.NormalisedFeatures[1]);
            }
        }
    }
}
