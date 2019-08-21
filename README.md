# Movie Review Sentiment

Movie Review Sentiment using [Blazor](https://dotnet.microsoft.com/apps/aspnet/web-apps/client) / [ConvNetSharp](https://github.com/cbovar/ConvNetSharp)

A Udacity Deep Learning NanoDegree exercise inspired me to do this.

### Projects:
* **Sentiment.Training**: Run to train the network. Model will be stored in *Model* folder (*sentiment.json*). Training data is located in *Data* folder (*reviews.txt* and *labels.txt*).
* **Sentiment.Inference**: Console App to test the inference. This code is reuinsg in Sentiment.Client.
* **Sentiment.Client**: Server-side Blazor app that allow user to classify review in real time.

This project requires [.NET Core 3.0](https://dotnet.microsoft.com/download/dotnet-core/3.0) (I am using 3.0.100 preview 8) and [Visual Studio Preview](https://visualstudio.microsoft.com/vs/preview/) (I am using 16.3.0).

![Screenshot](https://github.com/cbovar/MovieReviewSentiment/blob/master/img/Screen%20Shot.PNG)
