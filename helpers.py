import pandas as pd
import operator 

# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount 

# Use function to label datatype for training 1 or test 2 
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])
    
    # Prints counts by task and compare_dfcolumn for subset df
    #print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )
    
    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample of subset dataframe to create new df with subset values 
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(lambda x: x.sample(min(len(x), sampling_number), random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value
    
    # Prints counts by compare_dfcolumn for selected sample
    #print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    #print("\nSampled DF:\n",df_sampled)
    
    # Labels all datatype_var column as train_value which will be overwritten to 
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index: 
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    #print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon 
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered 
    
def train_test_dataframe(clean_df, random_seed=100):
    
    new_df = clean_df.copy()

    # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
    new_df.loc[:,'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)
    
    # creating a dictionary of categorical:numerical mappings for plagiarsm categories
    mapping = {0:'orig', 1:'train', 2:'test'} 

    # traversing through dataframe and replacing categorical data
    new_df.Datatype = [mapping[item] for item in new_df.Datatype] 

    return new_df


# Adds column in df 'Orig_idx' that contains the index of the original task for each answer file
def add_orig_loc(df):
    # Create list orig_idx that contains the original task index ordered A to E in the list
    orig_idx = df[(df['Category']==-1)].sort_values(by='Task').index.tolist()
    print("\n\nOriginal Task Indices A-E:",orig_idx,"\n")

    # Create task letter list a to e
    task_letter = ['a', 'b', 'c', 'd', 'e']

    # Add 'orig_task_idx' column to df
    df['Orig_idx'] = -1

    # Add correct index based upon orig_atoe_idx for each task letter
    for idx_orig in range(0, len(orig_idx)):
        # Create list of indices for a task based upon task letter and add to task_idx list
        task_idx = df[(df['Category']>-1) & (df['Task']==task_letter[idx_orig])].index.tolist()
        for idx_task in range(0,len(task_idx)):
            df.loc[task_idx[idx_task],'Orig_idx'] = orig_idx[idx_orig]