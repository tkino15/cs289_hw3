from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = 'data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************


def generate_design_matrix(filenames):
    texts = []
    for filename in tqdm(filenames):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read()
            except Exception as e:
                continue
            text = text.replace('\r\n', ' ')  # Remove newline character
            texts.append(text)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    vectorizer.fit(texts)

    design_matrix = vectorizer.transform(texts)

    return design_matrix.todense()


# ************** Script starts here **************
spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
n_spam = len(spam_filenames)
n_ham = len(ham_filenames)
n_test = len(test_filenames)
all_filenames = spam_filenames + ham_filenames + test_filenames
all_design_matrix = generate_design_matrix(all_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier

X = all_design_matrix[:(n_spam + n_ham)]
Y = np.array([1] * n_spam + [0] * n_ham).reshape((-1, 1))
test_design_matrix = all_design_matrix[(n_spam + n_ham):]

print(f'X:{X.shape}')
print(f'X:{test_design_matrix.shape}')

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat(BASE_DIR + 'my_spam_data.mat', file_dict)
