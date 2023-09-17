import os, glob
import numpy as np
import pandas as pd

class DataPreprocessor:
    "DataPreprocessor"

    def __init__(self,root_dir) -> None:
        self.root_dir = root_dir

    def getStepMixedFile(self,dirName,fileName):
        return dirName + 'awindaRecording_' + fileName.split('_')[1] + '.csv.stepMixed'
    

    def mapLabels(self,data,labels):
        k = 10
        data = pd.DataFrame(data)

        for index,row in labels.iterrows():
            data.loc[row[0]:row[0] + k, 'start'] = 1
            data.loc[row[1] - k :row[1], 'end'] = 1
        return data
    
    def takeStepMixesItem(self,item):
        return item[0]
    
    def sortLabeles(self,labeles):
        labeles = labeles.tolist()

        #Take all start indexes
        labeles.sort(key = self.takeStepMixesItem)
        return labeles
        
    
    def load_data(self):
        "Load all dataset from the root directory"
        dir_list = os.listdir(self.root_dir)
        dir_list = self.remove_testData(dir_list)
        
        result = []
        result_cleaned = []

        # Loop through the directory
        for dir in dir_list:
            dir = self.root_dir +'/'+ dir + '/' 
            # Get all csv files in [dir]
            files = glob.glob(dir + "*.csv")
            
            # Data 
            dir_labeled_data = []

            dir_labeled_cleaned_data = []
            
            #Iterate through the [files]
            for file in files:
                # read training data
                
                data = pd.read_csv(file)
                data['start'] = 0
                data['end'] = 0

                # read labels
                label_file = self.getStepMixedFile(dir,file.split('/')[3].split('.')[0])
                
                labels = pd.read_csv(label_file,header=None)

                labeled_data = self.mapLabels(data,labels)

                # Data Cleaning should happen herem to preserve indexes and labels
                cleaned_data = self.clean(labeled_data,labels)
                print(f"Cleaned data for {file}, {len(cleaned_data)}")

                # Append each set of data
                dir_labeled_data.append(labeled_data)
                # Append each cleaned set of data
                dir_labeled_cleaned_data.append(cleaned_data)
            
            # Concat dir_labeled_data, so 1 data frame per dir
            concat_data = pd.concat(dir_labeled_data).reset_index()

            concat_cleaned_data = pd.concat(dir_labeled_cleaned_data).reset_index()

            # Append data frames
            result.append(concat_data)
            result_cleaned.append(concat_cleaned_data)
        
        return (pd.concat(result).reset_index(drop=True),pd.concat(result_cleaned).reset_index(drop=True))
            
    def remove_testData(self, dir_list):
       # try:
        dir_list.remove('testdata.csv')
        #except:
        return dir_list
    
    def checkHasNextStep(self,labels, end_idx):
        has = False

        for label in labels.iterrows():
            start = label[1][0]
            if(start == end_idx):
                has = True
        return has
    
    def clean(self,labeled_data,labels):
        "Preform data cleaning per set"
        labeled_data = pd.DataFrame(labeled_data)

        # Sort the labeles
        labels = self.sortLabeles(labels.to_numpy())
        labels = pd.DataFrame(labels)

        # Start index
        start_idx = labels.iloc[0]
        start_idx = start_idx[0]

        # Last index
        last_idx = labels.iloc[-1]
        last_idx = last_idx[1]

        # Store all the indexes range to drop
        idx_to_drop = []

        for label in labels.iterrows():
            start = label[1][0]
            end = label[1][1]


            j = 1
            while self.checkHasNextStep(labels,end + j) != True:
                j = j+1
                if end + j > last_idx:
                    j = (len(labeled_data)) - (last_idx + 1)
                    break
            if j > 1:
                idx_range = [end + 1, end + j +1]
                idx_to_drop.append(idx_range)
        

        # Make a copy of labeled data

        labeled_data_deep_copy = []

        date_frame_copy = labeled_data.to_numpy()
        k = 0
        for idx_range in idx_to_drop:
            date_frame_copy = np.delete(date_frame_copy,slice(idx_range[0] - k, idx_range[1] - k),0)
            k = idx_range[1] - idx_range[0]


            labeled_data_deep_copy.append(date_frame_copy)
        
        # Remove start samples
        date_frame_copy = self.removeStartSamples(start_idx,date_frame_copy)
        return pd.DataFrame(date_frame_copy).reset_index(drop=True)
    
    def removeStartSamples(self,start_idx,labeled_data):
        l = np.delete(labeled_data,slice(0,start_idx),0)
        return pd.DataFrame(l)

