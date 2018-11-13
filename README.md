# Introduction to Artificial Intelligence: <br>John Salik, Eng. 
Written in Python, these examples are designed to follow-up on classroom teaching.  The introductory course is meant to provide a detailled overview of various AI algorithms following a methodology that relies on the intuition behind important mathematical formulations rather than a detailled treatment of the mathematics directly.  This allows for a pedagogy that trades off between blindly using off-the-shelf AI tools such as Tensorflow, and going deep into mathematical formulations that require advanced studies.

Students following this curriculum should know that these exercises are meant as incremental challenges, rather than explicit goals.  Advancement in the curriculum requires *self-study* and focused research in order to succeed in each challenge.  This said, students are expected to devote time into learning **Python** with the libraries that are implicitly used here.  It is expected that the instructor to provide the _theoretical background_  that relates to the examples given here.

### Self Study: How to do the laboratory work.
With the theoretical background provided by an instructor in a classroom setting, students should attend laboratory sessions with the goal of completing each work within a _two hour time-frame_, which can carry over to homework that should not exceed another _two hours_.  In this challenge-based pedagogy, students that cannot complete the work within a total of _four hours_ are having difficulty and must seek the assistance of the instructor in order to remained paced with weekly lectures.  **This is an intensive introductory course.**

In doing any of the laboratory work, students should follow the challenges _in the order presented_ as the sequence is meant to guide research.  The core of the challenges from an AI perspective will be relatively simple so the objective of doing the laboratory work is to understand **how to implement algorithms** and **how to apply them to solve problems.**  Once a student has completed the first challenge, they can proceed into the next.  If they cannot start the challenge _it is the responsibility of the **student** to seek the assistance of the instructor._  During the laboratory exercises, discussion between colleagues and the instructor is encouraged.  This focus on applied AI research is meant to generate questions that will ultimately produce a tangible result.  The _full, working examples_ provided here are meant to focus the student and assist them in producing their own work.

The body of laboratory work is based on proficient use of **Python 3**.  Through research, students should familiarize themselves with the following packages:

- **Mathematics:** numpy, scipy
- **Data Processing:** pandas
- **Visualization:** matplolib, seaborn
- **Machine Learning:** scikit-learn.

Of course, individual students may find that succeeding at a challenge may involve use of other packages.  This said, proficiency in using the above packages is a requirement to proceed with further courses in this series.

#### Students at Vanier College
Students in the _New Developments in Digital Technology_ course are responsible for the following:

- [x] **Lab 1:** Ex. 1
- [x] **Lab 2:** Ex. 1,2
- [x] **Lab 3:** Ex. 1,2
- [x] **Lab 4:** Ex. 1,2,3
- [x] **Lab 5:** EX. 1,3
- [ ] ~~**Lab 6**~~
- [ ] ~~**Lab 7**~~
- [ ] ~~**Lab 8**~~
- [ ] ~~**Lab 9**~~
- [ ] ~~**Lab 10**~~
- [ ] ~~**Lab 11**~~
- [ ] ~~**Lab 12**~~
- [ ] ~~**Lab 13**~~
- [ ] ~~**Lab 14**~~
- [ ] ~~**Lab 15**~~

Students should be prepared to produce projects and reports based on the knowledge gained from these incrementally difficult challenges.  The **theoretical base** acquired is of paramount concern here and Python implementation-details, while important - are __not__ the goal.  If the student has acquired a *language-independent* understanding of the the artificial intelligence systems described here, then they can consider themselves to have succeeded this course.

# Laboratory Work

### Lab 1: Working with Data
#### Exercise 1: Working with Tabular Data in a file.
> 1. Read data from a file and store it in memory.
> 2. Determine the dimensions of the data.
> 3. Perform a computation on the data.
> 4. Plot data based on computation.
#### Exercise 2: Synthesize Random 1D Test Data.
> 1. Write a function to generate 1D uniformly distributed data and plot a histogram with labeled axes.
> 2. Write a function to generate 1D normally distributed data and plot a histogram with labeled axes.
> 3. Simulate a noisy sine wave signal(uniform and Gaussian noise).
### Lab 2: Data Correlation
#### Exercise 1: Find a signal embedded in noise.
> 1. Read data from a file and process the stream in frames.
> 2. Using a sliding frame, compute a closeness score between the frame and a stored signal.
> 3. Determine the index within the data set that contains the start of the signal.
#### Exercise 2: Synthesize Random 2D Test Data.
> 1. Write a function that will generate a set of Gaussian random vectors with a a dictionary as input with parameters mean, variance, and size.
> 2. Generate several different sets of random vectors.
> 3. Make a scatter plot of the vectors.
### Lab 3: Regression, Model Fitting & Error Surfaces
#### Exercise 1: Create an error surface for a regression problem.
> 1. Create a simple linear model using fixed coefficients.
> 2. Create a new data set with additive Gaussian noise.
> 3. Compute the error, absolute error as well as the squared error. Plot them.
> 4. Estimate the linear (polynomial) coefficients for the noisy model.
> 5. Plot the error surface in a,b space identifying the minima.
#### Exercise 2: Given a data set, find the best fit model and make a prediction based on it.
> 1. Perform regression modeling of data against several models: (i)y=a*sin(x)+b,(ii)ln(y)=m*ln(x)+b,(iii)y=ax^3+b*x^2+c
> 2. Plot all three models with the data.
> 3. Compute residuals for each model.
> 4. Determine which of the models is the best fit for the given data.
> 5. Given a set of input values and the best fit model, predict the output.
### Lab 4: Adaptive processing using Error Gradients: Perceptron & ADALINE
#### Exercise 1: Train an ADALINE unit to separate data into two classes.  Use the learned decision line to classify new data.
> 1. Synthesize up a labelled 2D dataset with 250 points: (x,y,c) data separated into two target classes c = {+1, -1}
> 2. Plot data points for each class using the labels to distinguish points with color (in a subplot).
> 3. Set up 500 training cycles for an ADALINE with a learning rate of 1E-6.
> 4. Print the equation of the class separation line.  Optional: Normalize the equation.
> 5. Plot the class separation line using the weights determined from training (in a subplot).
> 6. Use the ADALINE to classify each point.  Superimpose the classification labels over the class separation line.
> 7. Compute the accuracy of the classifier.
#### Exercise 2: NLMS Noise Cancellation. Recover a signal that has been corrupted by noise using an adaptive filter.
> 1. Synthesize a pure sinusoid signal and corrupt it with noise (t={0..2pi}, f(x)= sin(5x).
> 2. Initialize the weight vector (w) for the filter (length k=500).
> 3. Initialize output and error arrays to hold these values for each shift of the filter through the data.
> 4. Recover the pure signal from the noisy data by adjusting filter weights adaptively using the NLMS learning rule. Store the output and error signals.
> 5. Plot the original signal, the learned noise signal, as well as the filter output.
#### EXERCISE 3: Using a sklearn's perceptron model, process sonar data for rock and mine data to distinguish between them.
> 1. Use pandas to read all data into an array and randomly shuffle the data.
> 2. Use the 80/20 rule. Put 80% of the raw data into two data vectors: Sonar and Classification (input and label). Use the remaining 20% of the data for testing the classifier.
> 3. Train _sklearn_'s **Perceptron()** unit that will adjust over a maximum of 1000 iterations with learning rate 1E-5.
> 4.  Determine what the training accuracy is using _sklearn_'s **accuracy_score()** function.
> 5.  Assume the 20% test data has just been read by the sonar.  Predict the classification using the perceptron and determine its classification accuracy.  Use sklearn's accuracy_score() function to do this.
> 6. From the perceptron, extract the weight vector.
> 7. Implement a function that will perform a classification based on an input and the extracted weights.
> 8. Construct a pandas table (dataframe) that will hold the following columns: test data, classification from the function-implemented perceptron and classification from sklean's perceptron classifier.
### Lab 5: The Multi-Layer Perceptron
#### Exercise 1: Demonstrate limitations of perceptrons: Learning 2-bit logic (AND, OR, XOR).
> 1. Create arrays that will hold inputs and corresponding outputs for each gate: AND, OR, XOR (truth tables).
> 2. To each input, add 500 gaussian random variables (0.2 variance).  There should now be 2000 points.
> 3. Create a target vector containing the target value representing the output for each logic gate.
> 4. Set up one perceptron for each logic gate.  Train it for a maximum number of 5000 iterations and a learning rate of 1E-6.
> 5. Print the learned truth table.
> 6. Plot the decision lines for the perceptrons trained on AND, OR, and XOR.
#### Exercise 2: Character recognition using a single layer of many perceptrons.
> 1. From sklearn, load the NIST handwritten dataset and display its associated description for the user.
> 2. Using 10 perceptrons from scikit and a "one-hot coding" scheme (1-of-10), train each to recognize one digit. Be sure to use the _entire_ dataset to train, not just data for a single digit.
> 3. Randomly select a image (of a digit) from the database.  Predict what it is.
> 4. Display the randomly selected image and the prediction of what digit it is.
> 5. Display a 10x10 image plot of some training data with the labeled randomly selected image.
> 6. Extract all of the perceptron weights and bias and store them in a text file with the bias in the last column.
> 7. Save the weights and bias in an Excel file with identifying columns (W0, W1, W2, ... W64, B). 
#### Exercise 3: Create a custom perceptron class and train it to distinguish between setosa and versicolor petals using length and width (Iris dataset).
> 1. Load the Iris Dataset from scikit.  Plot the sepal length vs sepal width for the setosa and versicolor flowers.
> 2. Assemble training data into an array with data point in rows (columns contain features).
> 3. Plot the data.
> 4. Create a perceptron class (**Percept**) with a constructor that will: initialize random weights (W), a maximum number of iterations (iterations=2000), learning rate (mu=1E-3), and a training score (score).  The constructor should take on optional parameters lrate and iter for learning rate and number of iterations respectively. Weights should be randomly initialized. Be sure to shuffle the data so that labeled data are in random positions.
> 5. Add a member function **train(X,T)** that takes data (X) and target labels (T) and implements the learning rule. Training should iterate over the entire data set using the set number of iterations.  It should return the training score.
> 6. Add a member function **predict(X)** that will return the output predictions given an input data vector.
> 7.  Instantiate the class using the default parameters.  Train the perceptron using the data and target classification.
> 8. Predict flower classifications for the training data.
> 9.  Using the perceptron weights, plot the decision line.
> 10. Save the plot.

### Lab 6: ~~Neural Networks & Backpropagation Training~~
### Lab 7: ~~Clustering & PCA~~
### Lab 8: ~~Data Trees~~ 
### Lab 9: ~~Markov Models 1~~
### Lab 10: ~~Markov Models 2~~
### Lab 11: ~~Markov Models 3~~
### Lab 11: ~~Genetic Algorithms 1~~
### Lab 11: ~~Genetic Algorithms 2~~
### Lab 14: ~~Genetic Algorithms & Neural Networks~~
### Lab 15: ~~Genetic Algorithms & Neural Networks~~

