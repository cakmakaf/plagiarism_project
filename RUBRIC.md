## Evaluation

Your project will be reviewed against the project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

When you are ready to submit your project, collect all of your project files -- all executed notebooks, and python files -- and compress them into a single zip archive for upload.

<a id='rubric'></a>
## Project Rubric

### Tests are Passed

#### Process a dataframe 
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  All unit tests are passed. |  Each non-modified test cell is executed and all tests are passed. |


### Data Pre-processing

#### Process a dataframe 
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Convert categorical to numerical data. |  Complete the `clean_dataframe` function such that it reads in a csv file and returns a dataframe with filenames, task types, and the associated plagiarism level (-1 to 3). |

#### Create a complete dataframe 
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Add `Text` and `Class` columns to an existing dataframe. |  Complete the function `complete_dataframe` such that it returns a dataframe with new columns for processed text and a class label for plagiarized (0) or not (1) or an original source text (-1). |

### Similarity Features

#### Create containment features
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Calculate the containment value for a file. |  Complete the `calculate_containment` function such that it accepts an n-gram size and file index and returns a normalized containment value for that file. |

#### Answer questions about containment and feature calculation

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| All questions about feature calculation are answered.  | All two questions about containment and features are answered and some explanation is given. |


#### Calculate the longest common subsequence

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Calculate the LCS between two files. |  Complete the `lcs_norm_word` function such that it takes in a source text and student answer text and returns a normalized LCS value for the longest common subsequence of words between the two texts. |


#### Create features
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a list of n-gram sizes to be used in feature calculations. | Define a range of n-gram sizes to use in calculating containment features. |

### Selecting "good" features

#### Select features to use in your model
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Select low-correlated features. |  Look at the correlation between features and select at least one feature to use in your final model, based on these correlation values. |

#### Answer question about feature selection

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Question about feature selection is answered.  | You've explained your reasoning behind your feature selection. |

### Modeling

#### Train and evaluate a model
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Train and evaluate a classification model. | Given your training and test features, train a model to label data as plagiarized or not. The `model_metrics` function should return the model's test accuracy and a confusion matrix; you should get above 90% classification accuracy. |

#### Answer question about model selection

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Question about model selection is answered.  | You've explained your reasoning behind your model selection. |
