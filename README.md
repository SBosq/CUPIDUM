# CUPIDUM Mobile App
---
<p>
  This idea came about as a result of a brainstorming session with several mates. I set out to first build a multiple choice questionnaire which would help us form clusters of compatible people/users.
  Since the target audience of this app was the student body of a religious university, we tried to make the questions reflect the values of said institution, not to mention that the
  questions needed to be in Spanish. Another issue that we faced was how we were going to verify that the every user was in fact enrolled in said university. How we went about this was
  making the register/login only accessible if the provided email has the correct university email format. Since we were using student email to verify student status, I decided to use this
  as a means to send a Google form with the questions. I tracked the amount of responses and even asked some friends to answer the questionnaire in order to have a solid control group dataset in order
  to properly train our Machine Learning model.
</p>
<hr>
<p>
  To be perfectly clear, some of the code that I used, was recycled from other projects that I have done before. However, other projects have been website related, or working on local environments only,
  this time around I was trying to build an <b><i>API using Python and Flask</i></b>. So how exactly did I get this to work? Well, I used <i>gspread</i> in order to handle the data extraction from the Google forms.
  In order to train the model and get the best results, I had to go some reading, digging around the documentation, and ultimately a couple of Google searches for a couple of things that I didn't quite understand.
  Finally, the libraries I ended up using apart from gspread, were <b><i>sklearn, pandas, matplotlib, seaborn, collections, and joblib</i></b>.
</p>
<hr>
<p>
  As for how I built the API and host it on a server, here are the steps:
  <ol>
    <li>First was generating credentials from Google Cloud API in order to access the Google Forms responses</li>
    <li>Accessing the responses via open_by_key</li>
    <li>From here the Google form responses are changed into a pandas Dataframe and deleting the timestamp since this isn't really needed for clustering compatible individuals</li>
    <li>Iterating through all the data in order to change all the string elements into numbers, this is done so the classifier can properly work</li>
    <li>Using MinMaxScaler, so that the numbers are all within a given range, in this case, the range is 0-1. From here, the new data is stored in a new Dataframe, keeping the same columns and index from the previous</li>
    <li>Before we can actually begin training the model, each individual email address is deleted since this also isn't needed for matchmaking</li>
    <li>Recall that our questionnaire was in Spanish, thus, we need to change the questions from Spanish to a single English word</li>
    <li>These words will now work as our model's features and we will use Principle Component Analysis (PCA) in order to reduce everything to just two variables</li>
    <li>Our <i>X</i> variable will have every answer or feature except for the individuals' sex; that belongs to our <i>Y</i> variable</li>
    <li>Once setup, PCA is initiated with two components and the results are then stored in a new Dataframe. As a final column, the individuals' sex is added and changed from 'Male' or 'Female' to a 0 or 1</li>
    <li>After all this is done, training the model can begin using train_test_split with a test size of 0.3 and a random state of 0 for future replications</li>
    <li>After some reading I decided to use <i>MiniBatchKMeans</i> with 3 clusters, this number was a result of a quick Elbow Method test, with the acquired dataset</li>
    <li>With the model trained and providing satisfactory results, the trained model is compressed into a "pickle" or .pkl format and is ready to be exported to the cloud server</li>
    <li>Finally each individual's answers are passed through the trained model and a cluster is predicted. Once the cluster is given, it is added to a final Dataset with the prepared data from before</li>
  </ol>
  This is still a brief description of how I created the API for this mobile application using Python and Flask. However, there are still other steps that were overlooked, such as how to create a Pythonanywhere account
  in order to host the API on the cloud. As well as how to code certain aspects of the API in order for it to give the correct response, and the usage of Postman for trial and error with certain responses or duplicated
  individual form responses.
</p>
