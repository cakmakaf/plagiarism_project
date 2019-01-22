from unittest.mock import MagicMock, patch
import sklearn.naive_bayes
import numpy as np
import pandas as pd
import re

# test csv file
TEST_CSV = 'data/test_info.csv'

class AssertTest(object):
    '''Defines general test behavior.'''
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])
    
    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message

def _print_success_message():
    print('Tests Passed!')

# test clean_dataframe
def test_clean_df(clean_dataframe):
    
    # test result
    clean_df = clean_dataframe(TEST_CSV)
                                
    # Check type
    assert isinstance(clean_df, pd.DataFrame), 'Returned type is {}.'.format(type(clean_df))
    
    # check columns
    assert list(clean_df) == ['File', 'Task', 'Category'], 'Unexpected column names: '+str(list(clean_df))
       
    # check conversion values
    assert clean_df.loc[0, 'Category'] == 1, '`heavy` plagiarism mapping test, failed.'
    assert clean_df.loc[2, 'Category'] == 0, '`non` plagiarism mapping test, failed.'
    assert clean_df.loc[30, 'Category'] == 3, '`cut` plagiarism mapping test, failed.'
    assert clean_df.loc[5, 'Category'] == 2, '`light` plagiarism mapping test, failed.'
    assert clean_df.loc[37, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    assert clean_df.loc[41, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    
    _print_success_message()
    
def test_complete_df(clean_df, complete_dataframe):
    
    complete_df = complete_dataframe(clean_df, clean_df['File'])
    
    # Check type
    assert isinstance(complete_df, pd.DataFrame), 'Returned type is {}.'.format(type(complete_df))
    
    # check additional columns
    assert len(complete_df['Text']) != 0, 'Text column, not found.'
    assert len(complete_df['Class']) != 0, 'Class column, not found.'    

    # check shape
    assert complete_df.shape[0]==100, 'Incorrect number of rows, expecting 100, got '+ str(complete_df.shape[0])
    assert complete_df.shape[1]==5, 'Incorrect number of columns, expecting 5, got '+ str(complete_df.shape[1])

    # check class values
    for idx in range(0, len(complete_df['Category'])):
        if(complete_df['Category'][idx] > 1): 
            assert complete_df['Class'][idx] == 1, \
            'Plagiarized cases should have a Class=1.'
        else:
            assert complete_df['Class'][idx] == complete_df['Category'][idx], \
            'Non-plagiarized cases should have the same Category and Class value.'
    
    # check text processing
    assert complete_df.loc[1, 'Text'].islower(), 'Text should be all lowercase.'    

    _print_success_message()

def test_containment(complete_df, containment_fn):
    
    # check basic format and value 
    test_val = containment_fn(complete_df, n=1, file_index=0)
    
    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)
    
    # known vals for first few files
    ngram_1 = [0.3617021276595745, 1.0, 0.8487394957983193, 0.5225225225225225]
    ngram_3 = [0.00975609756097561, 0.9635416666666666, 0.6084905660377359, 0.15934065934065933]
    
    # results for comparison
    results_1gram = []
    results_3gram = []
    
    for i in range(4):
        val_1 = containment_fn(complete_df, n=1, file_index=i)
        val_3 = containment_fn(complete_df, n=3, file_index=i)
        results_1gram.append(val_1)
        results_3gram.append(val_3)
        
    # check correct results
    assert all(np.isclose(results_1gram, ngram_1, rtol=1e-05)), \
    'n=1 calculations are incorrect. Double check the intersection calculation.'
    # check correct results
    assert all(np.isclose(results_3gram, ngram_3, rtol=1e-05)), \
    'n=3 calculations are incorrect.'
    
    _print_success_message()
    
def test_lcs(df, lcs_word):
    
    # check basic format and value 
    source_text = df.loc[df.loc[10,'Orig_idx'], 'Text']
    answer_text = df.loc[10, 'Text']
    test_val = lcs_word(source_text, answer_text)
    
    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)
    
    # known vals for first few files
    lcs_vals = [0.1917808219178082, 0.8207547169811321, 0.8464912280701754, 0.3160621761658031, 0.24257425742574257]
    
    # results for comparison
    results = []
    
    for i in range(5):
        source_text = df.loc[df.loc[i,'Orig_idx'], 'Text']
        answer_text = df.loc[i, 'Text']
        # calc lcs
        val = lcs_word(source_text, answer_text)
        results.append(val)
        
    # check correct results
    assert all(np.isclose(results, lcs_vals, rtol=1e-06)), 'LCS calculations are incorrect.'
    
    _print_success_message()
    
def test_selection(features_df, select_features):
    
    test_selection = ['c_1', 'lcs_word']
    
    train_data, test_data = select_features(features_df, test_selection)
    
    # Check types
    assert isinstance(train_data, tuple),\
        'train_data is not a tuple, instead got type: {}'.format(type(train_data))
    assert isinstance(test_data, tuple),\
        'test_data is not a tuple, instead got type: {}'.format(type(test_data))
        
    (train_x, train_y) = train_data
    (test_x, test_y) = test_data
    
    # Check types
    assert isinstance(train_x, pd.DataFrame),\
        'train_x is not a DataFrame, instead got type: {}'.format(type(train_x))
    assert isinstance(train_y, np.ndarray),\
        'test_y is not an array of values, instead got type: {}'.format(type(train_y))
    
    assert isinstance(test_x, pd.DataFrame),\
        'train_x is not a DataFrame, instead got type: {}'.format(type(test_x))
    assert isinstance(test_y, np.ndarray),\
        'test_y is not an array of values, instead got type: {}'.format(type(test_y))
        
    # should hold all 95 submission files
    assert len(train_x) + len(test_x) == 95, \
        'Unexpected amount of train + test data. Expecting 95 submission files, got ' +str(len(train_x) + len(test_x))
    
    _print_success_message()
    
def test_model_metrics(model_metrics):
    
    train_data = np.array([np.random.rand(10)]*2).T
    train_x = pd.DataFrame(train_data, index=range(10), columns=['c_1', 'lcs_word'])
    train_y = np.random.randint(2, size=10) # 0 or 1
    
    test_data = np.array([np.random.rand(5)]*2).T
    test_x = pd.DataFrame(test_data, index=range(5), columns=['c_1', 'lcs_word'])
    test_y = np.random.randint(2, size=5) # 0 or 1
    
    test_model = sklearn.naive_bayes.MultinomialNB() # benchmark, "bad" model
    
    # test function
    acc, c_matrix = model_metrics(test_model, train_x, train_y, test_x, test_y)
    
    # Check types
    assert isinstance(acc, float),\
        'accuracy is not a float.'
    assert isinstance(c_matrix, np.ndarray),\
        'confusion_matrix is not an array.'
    
    assert acc<=1.0, 'Accuracy is larger than 1.'    
    
    _print_success_message()
    
        