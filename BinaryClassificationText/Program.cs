using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace BinaryClassificationText
{

    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _dataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "yelp_labelled.txt");
        private static string _dataPredPath => Path.Combine(_appPath, "..", "..", "..", "Data", "all_wrong.tsv");

        private static string _csvPath => Path.Combine(_appPath, "..", "..", "..", "Data", "data2.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            TrainTestData splitDataView = LoadData(_mlContext);

            ITransformer model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

            Evaluate(_mlContext, model, splitDataView.TestSet);

            SaveModelAsFile(_mlContext, splitDataView.TrainSet.Schema, model);

            //UseModelWithSingleItem(_mlContext, model);

            UseModelWithData(_mlContext, model);
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            
            IDataView dataView = mlContext.Data.LoadFromTextFile<EcologyIssue>(_csvPath, hasHeader: false);


            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.4);
           
            return splitDataView;
        }


        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(EcologyIssue.SentimentText))
            
             .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
           
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);

            Console.WriteLine("=============== End of training ===============");

            return model;
        }

        //вычисляет метрики качества для модели на основе указанного набора данных
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            IDataView predictions = model.Transform(splitTestSet);
            
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

           
        }

        //private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        //{
        //    var predictionFunction = mlContext.Model.CreatePredictionEngine<EcologyIssue, IssuePrediction>(model);

        //    var flag = true;

        //    while (flag)
        //    {
        //        Console.WriteLine("введите текст для проверки");

        //        var newSample = Console.ReadLine();
        //        EcologyIssue sampleStatement = new EcologyIssue
        //        {
        //            SentimentText = newSample
        //        };

        //        var resultPrediction = predictionFunction.Predict(sampleStatement);

        //        Console.WriteLine();
        //        Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

        //        Console.WriteLine();
        //        Console.WriteLine($"Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

        //        Console.WriteLine("=============== End of Predictions ===============");
        //        Console.WriteLine();
        //    }
        //}

        private static void UseModelWithData(MLContext mlContext, ITransformer model)
        {
            //загрузка данных из проверочного файла
            IDataView dataNewPredView = mlContext.Data.LoadFromTextFile<EcologyIssue>(_dataPredPath, hasHeader: false);
            //IDataView predictions = model.Transform(dataNewPredView);

            //для оценки каждого текста
            //var predictionFunction = mlContext.Model.CreatePredictionEngine<EcologyIssue, IssuePrediction>(model);

            IDataView predictions = model.Transform(dataNewPredView);
            var predictedResults = mlContext.Data.CreateEnumerable<IssuePrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            Console.WriteLine();

            foreach (IssuePrediction prediction in predictedResults)
            {
                Console.WriteLine($"Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");




            //StreamReader sr = new StreamReader(_dataNewPredPath);

            //List<EcologyIssue> sentiments = new List<EcologyIssue> { };
            //foreach (EcologyIssue prediction in predictedResults)
            //{
            //    var line = sr.ReadLine();
            //    var sent = new EcologyIssue { SentimentText = line };
            //    var res = predictionFunction.Predict(sent);
            //    Console.WriteLine($"Sentiment: {res.SentimentText} | Prediction: {(Convert.ToBoolean(res.Prediction) ? "Positive" : "Negative")} | Probability: {res.Probability} ");
            //}
        }
        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
          
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
