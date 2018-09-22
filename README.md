# Introduction to Artificial Intelligence: John Salik
Written in Python, these examples are designed to follow-up on classroom teaching.  The introductory course is meant to provide a detailled overview of various AI algorithms following a methodology that relies on the intuition behind important mathematical formulations rather than a detailled treatment of the mathematics directly.  This allows for a pedagogy that trades off between blindly using off-the-shelf AI tools such as Tensorflow, and going deep into mathematical formulations that require advanced studies.
## 

### Lab 1
#### Exercise 1: Working with Tabular Data in a file.
1. [ ] Read data from a file and store it in memory.
2. [ ] Determine the dimensions of the data.
3. [ ] Perform a computation on the data.
4. [ ] Plot data based on computation.
#### Exercise 2: Synthesize Random 1D Test Data.
1. [ ] Write a function to generate 1D uniformly distributed data and plot a histogram with labeled axes.
2. [ ] Write a function to generate 1D normally distributed data and plot a histogram with labeled axes.
3. [ ] Simulate a noisy sine wave signal(uniform and Gaussian noise).
### Lab 2 
#### Exercise 1: Find a signal embedded in noise.
1. Read data from a file and process the stream in frames.
2. Using a sliding frame, compute a closeness score between the frame and a stored signal.
3. Determine the index within the data set that contains the start of the signal.
#### Exercise 2: Synthesize Random 2D Test Data.
1. Write a function that will generate a set of Gaussian random vectors with a a dictionary as input with parameters mean, variance, and size.
2. Generate several different sets of random vectors.
3. Make a scatter plot of the vectors.
### Lab 3
#### Exercise 1: Create an error surface for a regression problem.
1. Create a simple linear model using fixed coefficients.
2. Create a new data set with additive Gaussian noise.
3. Compute the error, absolute error as well as the squared error. Plot them.
4. Estimate the linear (polynomial) coefficients for the noisy model.
5. Plot the error surface in a,b space identifying the minima.
#### Exercise 2: Given a data set, find the best fit model and make a prediction based on it.
1. Perform regression modeling of data against several models: (i)y=a*sin(x)+b,(ii)ln(y)=m*ln(x)+b,(iii)y=ax^3+b*x^2+c
2. Plot all three models with the data.
3. Compute residuals for each model.
4. Determine which of the models is the best fit for the given data.
5. Given a set of input values and the best fit model, predict the output.
### Lab 4
#### Exercise 1: Train a perceptron to separate data into two classes.  Use the learned decision line to classify new data.
#### Exercise 2: Use the LMS algorithm in a simulation of active noise cancellation.
