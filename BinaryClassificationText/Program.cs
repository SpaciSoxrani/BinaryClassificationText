using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace BinaryClassificationText
{

    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");

        private static string _dataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "yelp_labelled.txt");

        private static MLContext _mlContext;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            TrainTestData splitDataView = LoadData(_mlContext);

            ITransformer model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

            Evaluate(_mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(_mlContext, model);
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            
            IDataView dataView = mlContext.Data.LoadFromTextFile<EcologyIssue>(_dataPath, hasHeader: false);
           
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
           
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

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<EcologyIssue, IssuePrediction>(model);

            EcologyIssue sampleStatement = new EcologyIssue
            {
                SentimentText = "Одним из первых Жениных проектов было обновление БД и бэкенда для ленты новостей. Другим — хранение и доставка уведомлений о важных событиях в шапке сайта, архитектуру которого он написал с нуля. Благодаря усилиям Жени все картинки в VK хранятся максимально компактно — это самый большой проект, которым он занимался. В прошлом году он возглавил команду, в которую когда-то пришел, и стал отличным руководителем, — передаёт слова коллег ребёнка E1.RU."
            };
            
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
    }
}
