# Fake-News-Detection
Final Project for Machine Learning Class at University of Delaware.
Please find the Datasets ([Here](https://drive.google.com/drive/folders/1T_bZbMCHlW-EDxTdGNw3IPsu-XZp-mbG?usp=drive_link)) 


## Project Overview
A binary classifier that distinguishes real from fake news articles. Key components:
- Data Loading & Labeling — Combine two CSVs, assign 1 to real, 0 to fake.
- Text Preprocessing — Lowercasing, regex-based cleaning, NLTK stopword removal, Snowball stemming.
- Feature Extraction — TF-IDF vectorization to transform text into high-dimensional sparse vectors.
- Modeling — Train SVM with linear and Gaussian (RBF) kernels; tune C and gamma.
- Evaluation — Report accuracy and full precision/recall/F₁ metrics on train/test splits.

`
git clone https://github.com/rushabhdhoke/Fake-News-Detection-Final-Project.git  
cd Fake-News-Detection-Final-Project  
python -m venv venv && source venv/bin/activate  # optional but recommended  
pip install -r requirements.txt
`

### Data
- fake.csv: ~23.5 K fake news articles
- true.csv: ~21.4 K real news articles

Both files contain title, text, subject, date columns. We drop date after concatenation for simplicity.


### Preprocessing
1. Lowercasing & Cleaning
Strip URLs via regex: r"http\S+|www\S+"
Remove non-alphabetic chars: r"[^a-zA-Z\s]" 
Convert to lowercase for normalization.

2. Stopword Removal
Use NLTK’s English stopword list to drop high-frequency tokens (e.g., “and”, “the”)

3. Stemming
Apply SnowballStemmer('english') to reduce words to their root form (e.g., “running” → “run”) 


### Feature Extraction
#### TF-IDF Vectorization
- Instantiate TfidfVectorizer(), fit on concatenated "title" + " " + "text" corpus.
- Converts text into a sparse (n_samples × n_features) matrix, weighting terms by inverse document frequency to emphasize distinctive words


### Modeling
#### Support Vector Machine (SVM)
- Linear Kernel: SVC(kernel='linear', C=1.0) for a baseline in high-dimensional TF-IDF space
- RBF Kernel: SVC(kernel='rbf', gamma=γ, C=C) to capture non-linear patterns.

#### Hyperparameter Tuning
- Sweep C (0.01 → 100) to adjust regularization strength.
- Sweep gamma (0.001 → 10) for RBF width.
- Plot Accuracy vs. C and Accuracy vs. Gamma on log scales to select optimal values.

### Evaluation
- Split data: train_test_split(test_size=0.2, random_state=42).
- Compute Accuracy, then detailed metrics via classification_report(y_true, y_pred) (precision, recall, F₁ for each class) 
- Compare training vs. test performance to check for overfitting.

### Results
- Best Linear SVM: Test Accuracy ≈ X %
- Best RBF SVM: Test Accuracy ≈ Y %
- Full precision/recall/F₁ breakdown in Final_Project.ipynb.


### Contributing
Feel free to fork, improve preprocessing (e.g., add lemmatization), experiment with ensemble models, or integrate deep-learning classifiers.

-- Submit PRs and issues on GitHub.

