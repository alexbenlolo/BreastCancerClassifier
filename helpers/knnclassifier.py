from classifier import *

class knnclassifier(classifier):
    def __init__(self, K = 1):
        self.k = K

    def train(self, train_data):
        '''Train the classifier with a list of data_item objects.'''
        self.data = train_data.copy()

    def predict(self, data_point):
        '''Predict the class of 'data_point' by majority vote of the 'k' 
        closest points in the 'training' dataset.'''
        
        nearest = sorted(self.data, key=lambda x: distance(data_point, x.data))
        
        # now we have all items sorted by increasing distance.
        
        counts = {}
        for ld in nearest[:self.k]:
            counts[ld.label] = counts.get(ld.label, 0) + 1
        return max(counts.keys(), key=lambda key: counts[key])
    

if __name__ == "__main__":
    from datasets import read_datasets

    datasets = read_datasets()
    for name in sorted(datasets):
        dataset = datasets[name]
        print(name)
        # Create a normalized dataset.
        #
        norm_dataset = normalize_dataset(dataset)

        print("k_nearest:")
        for k in range(1, 14, 2):
            pct = evaluate(dataset, knnclassifier, 4, K=k)
            print('K {}: {:.2%}'.format(k, pct))
        print("k_nearest, normalized:")
        for k in range(1, 14, 2):
            pct = evaluate(norm_dataset, knnclassifier, 4, K=k)
            print('K {}: {:.2%}'.format(k, pct))
