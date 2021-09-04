using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassificationText
{
    public class EcologyIssue
    {
        [LoadColumn(0)]
    public string SentimentText;

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment;
    }

    //используется для прогнозирования после обучения модели.
    public class IssuePrediction : EcologyIssue
    {
        //cтолбец для прогнозирования

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
