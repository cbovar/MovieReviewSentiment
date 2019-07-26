# Movie Review Sentiment

Movie Review Sentiment using Blazor / ConvnetSharp

### Projects:
* **Sentiment.Training**: Run to train the network. Model will be stored in *Model* folder (*sentiment.json*). Training data are located in *Data* folder (*reviews.txt* and *labels.txt*).
* **Sentiment.Inference**: Console App to test the inference. This code is reuinsg in Sentiment.Client.
* **Sentiment.Client**: Server-side Blazor app that allow user to classify review in real time

This project requires .Net Core 3.0 (I am using 3.0.100 preview 7) and Visual Studio Preview (I am using 16.3.0).

![Screenshot](https://github.com/cbovar/MovieReviewSentiment/blob/master/img/Screen%20Shot.PNG)
