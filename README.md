# Bayesian-Classification-Machine-Learning
Naïve Bayesian classifier that can read a movie review and decide, if the review author would rate the movie as positive or negativebased on the text entered.
Machine Learning Assignment 
# Task 1
For task 1 Itook in the excel file and mapped thestring values for sentiment and split to integers.I split thedata for the split value and also the sentiment.
# Task 2
This functiontook in the training data and took the column review. It took only the alphanumeric characters, made them lowercase and split them.I then looped through each review then each word. I checked if the length of the word was greater than the minimum word length. It would create a list and set the value of the word as 1, but if it was already in the list it would at 1.
# Task 3
This task calls taskuses the inputs from its own parameters to call task 2 so we have the total word occurrences. It then splits them between positive and negative.For the next part Icreated an extra method to cut down on repeated code. This method strips the reviews like before.1 loop iterates through every review. It then checks every word in the word occurrenceslist. If the word is in the review it is added to a list like before.I then looped through the word occurrencesto check and see what words did not appear in a review and made that value 0.
# Task 4
Thistask goes through every word in word occurrences. It will calculatea likelihood by checking the number of reviews it appears in, then divide it by the sum of values for that sentiment. It uses laplace smoothing also.The priors are calculated by dividing the length of total reviews for that sentiment divided by the length of total reviews.
# Task 5
For task 5 I followed a youtube tutorial for calculating NaïveBayes. An input is taken in and the strings are split. The prior is then multiplied by the probability of each word occurring. A value 1 or 0 is returned depending on the result. This does work.My code using the logs was not steady and did not function correctly 100% of the time so I have it commented out.
# Task 6
For this implementation I loaded up the data again. Select the model Stratified KFold with 3 splits.We loop 1-10 for different K values where k is the minimum word length when we call task 2 again.We use loop through the splits and call task 5 which checks each review. The predicted labels are then created.We use a confusion matrix to determine whether this was accurate. An accuracy score is also gathered.The accuracy score for each k value is put in a list and then the mean is calculated from it.At the end we print off the means to determine which is the most accurate. From our output we determine 5 is the optimal minimum word occuren
