import pandas as pd
import re
import json
from pandas.api.types import is_number

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


import multiprocess as mp
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open('keywords.json', 'r', encoding='utf-8') as f:
    KEYWORDS = json.load(f)

def clean_field(val):
    try:
        if pd.isna(val) or val == '':
            return ''
        return str(val)
    except Exception as e:
        print(f"[ERROR] clean_field failed for value: {val}, error: {e}")
        return ''

def clean_city(val):
    try:
        if pd.isna(val) or val == '':
            return pd.NA
        val = str(val).lower()
        val = re.sub(r'[^a-z0-9\s]', ' ', val)
        val = re.sub(r'\s+', ' ', val)
        return val.strip()
    except Exception as e:
        print(f"[ERROR] clean_city failed for value: {val}, error: {e}")
        return pd.NA

def clean_pincode(val):
    try:
        if pd.isna(val):
            return pd.NA
        if is_number(val) == True:
            val = int(val)  # check if it's a valid number
        val = str(val)
        if len(val) != 6:
            return pd.NA
        return val
    except Exception as e:
        print(f"[ERROR] clean_pincode failed for value: {val}, error: {e}")
        return pd.NA

def clean_address(address):
    try:
        if pd.isna(address):
            return pd.NA
        address = str(address).lower()
        address = re.sub(r'[^a-z0-9\s]', ' ', address)
        address = re.sub(r'\b\d{6,}\b', '', address)
        address = re.sub(r'address line \d', '', address)
        address = re.sub(r'nullnull|nan', '', address)
        address = re.sub(r'null|nan', '', address)
        address = re.sub(r'\s+', ' ', address).strip()
        if address == '':
            return pd.NA
        return address
    except Exception as e:
        print(f"[ERROR] clean_address failed for value: {address}, error: {e}")
        return pd.NA

def clean_and_validate_address(address):
    try:
        main_address = address
        address = clean_address(address)
        if pd.isna(address):
            return False
        if len(address.split()) <= 1:
            return False
        if len(address) < 15:
            if isinstance(main_address, str) and main_address.isupper() and main_address.isalpha():
                return True
            else:
                return False
        if len(address.split()) <= 6 and not re.search(r'\d', address):
            for kw in KEYWORDS:
                if kw in address:
                    return True
            return False
        return True
    except Exception as e:
        print(f"[ERROR] clean_and_validate_address failed for: {address}, error: {e}")
        return False

def make_others(data, target_col, min_samples=400, max_samples=None, valid_cps=None):
    try:
        if valid_cps is None:
            cp_counts = data[target_col].value_counts()
            valid_cps = cp_counts[cp_counts >= min_samples].index.tolist()
            print(f"[INFO] Found {len(valid_cps)} valid classes with >= {min_samples} samples")

        removed_df = data[~data[target_col].isin(valid_cps)]
        removed_cps = list(set(removed_df[target_col]))
        print(f"[INFO] {len(removed_df)} samples reassigned to 'OTHER'")

        data.loc[removed_df.index, target_col] = "OTHER"
        return data, valid_cps + ['OTHER'] , removed_cps
    except Exception as e:
        print(f"[ERROR] make_others failed, error: {e}")
        return data, valid_cps or []

def data_preprocess(data, feature_col, target_col, min_samples=None, max_samples=None, valid_cps=None):
    try:

        print("[INFO] Starting preprocessing...")

        print(f'Data rows{len(data)} at start')


        # Removing V series and Branches
        before_drop = len(data)
        data = data[data['Delivered_By_Hub_Type']=='FR']
        data = data[data['v_series']==0]
        after_drop = len(data)
        print(f"[INFO] Dropped {before_drop - after_drop} v series and Branches")



    #     # Clean address fields
    #     data['address'] = (
    #         data['street_1'].apply(clean_field) + ' ' +
    #         data['street_2'].apply(clean_field) + ' ' +
    #         data['street_3'].apply(clean_field)
    #     ).str.strip()
    #     print("[INFO] Address fields concatenated")

    #     # Clean other fields
    #     data['city'] = data['city'].apply(clean_city)
    #     data['pincode'] = data['pincode'].apply(clean_pincode)
    #     data['address'] = data['address'].apply(clean_address)

        # cps = pd.read_csv('data/input/uq_cp_codes.csv')
        # data=pd.merge(data,cps,left_on='Delivered_By_Hub_Code',right_on='0',how='left')


        # Drop NA
        before_drop = len(data)
        data.dropna(subset=['address', 'foc_receiver_pincode', 'foc_receiver_city', target_col,'0'], inplace=True)
        after_drop = len(data)
        print(f"[INFO] Dropped {before_drop - after_drop} rows with NA in essential columns")

        # Remove duplicates
        before_dup = len(data)
        data.drop_duplicates(subset=['full_address', target_col], inplace=True)
        # data.drop_duplicates(subset=['address','Delivered_By_Hub_Code'], inplace=True)
        after_dup = len(data)
        print(f"[INFO] Dropped {before_dup - after_dup} duplicate address-target rows")

        # data = clean_V_cp(data)






    #     # Validation
    #     data['is_valid'] = data['address'].apply(clean_and_validate_address)
    #     print(f"[INFO] Completed validation on addresses")

    #     # Final feature column
    #     data[feature_col] = data['address'] + ' ' + data['pincode'] + ' ' + data['city']
        
    #     # Set invalid targets to INVALID
    #     invalid_count = (~data['is_valid']).sum()
    #     data.loc[~data['is_valid'], target_col] = "INVALID"
    #     print(f"[INFO] Marked {invalid_count} rows as INVALID")

        # Grouping low-sample targets
        # data, valid_cps = make_others(data, target_col, min_samples, max_samples, valid_cps)
        print(f"[INFO] Finished processing. Final rows: {len(data)}")
        
        #data['cp_code'] = data[target_col]

        return data, valid_cps
    except Exception as e:
        print(f"[ERROR] data_preprocess failed, error: {e}")
        return data, valid_cps or []






class AddressClassifier:
    """
    A text classification pipeline for address-based CP Code prediction.
    Handles vectorization, model training, evaluation, and predictions.
    """

    def __init__(
        self,
        train_data=None,
        val_data=None,
        train_feature=None,
        train_target=None,
        encoder=None,
        model=None,
        vectorizer=None,
        directory=None
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.train_feature = train_feature
        self.train_target = train_target
        self.encoder = encoder
        self.model = model
        self.vectorizer = vectorizer

        self.directory = directory

        self.model_path = f"{self.directory}/address_model.joblib"
        self.vectorizer_path = f"{self.directory}/address_vectorizer.joblib"
        self.encoder_path = f"{self.directory}/address_encoder.joblib"
        self.report_path = f"{self.directory}/val_accuracy_report.csv"


        self.print_model_paths()


    def print_model_paths(self):
        """Print all model, vectorizer, and encoder paths."""
        print(f"Model path: {self.model_path}")
        print(f"Vectorizer path: {self.vectorizer_path}")
        print(f"Encoder path: {self.encoder_path}")


    def _vectorize_data(self):
        """Fit vectorizer on train data and transform both train & test sets."""
        print("[INFO] Vectorizing data...")
        X_train_vec = self.vectorizer.fit_transform(self.train_data[self.train_feature])
        X_val_vec = self.vectorizer.transform(self.val_data[self.train_feature])
        print(f"[INFO] Train vectorized shape: {X_train_vec.shape}")
        print(f"[INFO] Val vectorized shape: {X_val_vec.shape}")
        return X_train_vec, X_val_vec

    def _load_model_if_needed(self):
        """Load model and vectorizer from disk if not already loaded."""
        if self.model is None and self.model_path:
            print(f"[INFO] Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
        if self.vectorizer is None and self.vectorizer_path:
            print(f"[INFO] Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)

    def train_model(self):
        """
        Train the model using provided training data.
        Supports partial_fit if available for large datasets.
        """
        # self._load_model_if_needed()

        X_train_vec, X_val_vec = self._vectorize_data()
        y_train = self.train_data[self.train_target]
        y_val = self.val_data[self.train_target]

        self.model.fit(X_train_vec, y_train)

        print("[INFO] Evaluating model...")
        y_val_pred = self.model.predict(X_val_vec)
        y_val_pred_str = self.encoder.inverse_transform(y_val_pred)
        y_val_str = self.encoder.inverse_transform(y_val)

        # Save classification report to a DataFrame
        from sklearn.metrics import classification_report
        import pandas as pd

        report_dict = classification_report(y_val_str, y_val_pred_str, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.index.name = "class"
        report_df.to_csv(self.report_path)

        # print(classification_report(y_val_str, y_val_pred_str))

    def test_predict(self, test_data, test_feature="full_address", test_target="CP_Code_Actually_Delivering"):
        """
        Predict top-2 classes for given test data.
        Adds predictions and probabilities to the dataframe.
        """
        self._load_model_if_needed()

        print("[INFO] Transforming test data...")
        X_test_vec = self.vectorizer.transform(test_data[test_feature])

        print("[INFO] Generating predictions...")
        y_proba = self.model.predict_proba(X_test_vec)
        top2_indices = np.argsort(y_proba, axis=1)[:, -2:][:, ::-1]

        y_pred1 = self.encoder.inverse_transform(top2_indices[:, 0])
        y_pred2 = self.encoder.inverse_transform(top2_indices[:, 1])
        max_proba = np.max(y_proba, axis=1)

        print("[INFO] Classification Report (Top-1 Predictions):")
        print(classification_report(test_data[test_target], y_pred1))

        test_data["predicted_cp_code_ml1"] = y_pred1
        test_data["predicted_cp_code_ml2"] = y_pred2
        test_data["predicted_cp_code_ml_proba"] = max_proba
        test_data["ml_pred_match"] = test_data[test_target] == y_pred1
        test_data["ml_pred_match2"] = test_data[test_target] == y_pred2

        return test_data

    def save_model(self, model_path=None, vectorizer_path=None, encoder_path=None):
        """Save model, vectorizer, and encoder to disk."""
        model_path = model_path or self.model_path
        vectorizer_path = vectorizer_path or self.vectorizer_path
        encoder_path = encoder_path or self.encoder_path

        print(
            model_path, vectorizer_path, encoder_path
        )
        if model_path:
            joblib.dump(self.model, model_path)
            print(f"[INFO] Model saved to {model_path}")
        if vectorizer_path:
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"[INFO] Vectorizer saved to {vectorizer_path}")
        if encoder_path:
            joblib.dump(self.encoder, encoder_path)
            print(f"[INFO] Encoder saved to {encoder_path}")

    def load_model(self, model_path=None, vectorizer_path=None, encoder_path=None):
        """Explicitly load model, vectorizer, and encoder from disk."""
        self.model = joblib.load(model_path or self.model_path)
        self.vectorizer = joblib.load(vectorizer_path or self.vectorizer_path)
        if encoder_path:
            self.encoder = joblib.load(encoder_path)
        print("[INFO] Model, vectorizer, and encoder loaded.")



class AddressClassifierInference:
    """
    Inference-only pipeline for address-based CP Code prediction.
    Loads a trained model, vectorizer, and encoder for predictions.
    """

    def __init__(self, directory):
        """
        Initialize the inference class by loading model artifacts.
        """


        self.directory = directory

        self.model_path = f"{self.directory}/address_model.joblib"
        self.vectorizer_path = f"{self.directory}/address_vectorizer.joblib"
        self.encoder_path = f"{self.directory}/address_encoder.joblib"
        self.report_path = f"{self.directory}/test_acurracy_report.csv"
        self.test_accuracy_data_path = f"{self.directory}/test_accuracy_data.parquet"
        self.other_cp_path = f"{self.directory}/other_cp_path.csv"
        self.trained_cp_path = f"{self.directory}/trained_cp_path.csv"

        

        print("[INFO] Loading model...")
        self.model = joblib.load(self.model_path)

        print("[INFO] Loading vectorizer...")
        self.vectorizer = joblib.load(self.vectorizer_path)

        print("[INFO] Loading encoder...")
        self.encoder = joblib.load(self.encoder_path)


    def predict(self, data, feature_col="full_address", target_col=None, top_k=2):
        """
        Predict top-k classes for given dataframe.

        Args:
            data (pd.DataFrame): Input dataframe containing feature_col (and optionally target_col for evaluation)
            feature_col (str): Column name with address text
            target_col (str): Column name with true CP codes (optional)
            top_k (int): Number of top predictions to return

        Returns:
            pd.DataFrame: Dataframe with added prediction columns
        """
        print("[INFO] Vectorizing input data...")
        X_vec = self.vectorizer.transform(data[feature_col])

        print(f"[INFO] Generating top-{top_k} predictions...")
        y_proba = self.model.predict_proba(X_vec)
        top_indices = np.argsort(y_proba, axis=1)[:, -top_k:][:, ::-1]

        for i in range(top_k):
            data[f"predicted_cp_code_ml{i+1}"] = self.encoder.inverse_transform(top_indices[:, i])

        data["predicted_cp_code_ml_proba"] = np.max(y_proba, axis=1)


        

        # Optional evaluation
        if target_col is not None:
            print("[INFO] Evaluation (Top-1 predictions):")
            # Get classification report as dict
            report_dict = classification_report(
                data[target_col], 
                data["predicted_cp_code_ml1"], 
                output_dict=True
            )
            # Convert to DataFrame
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.index.name = "class"
            report_df.to_csv(self.report_path)

        return data



def city_models(city, data, test_data):

    MODEL_DIRECTORY = f"models/{city}"

    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    # ============================
    # 2. Define Encoding
    # ============================

    feature_col = "full_address"
    target_col = "cp_code"

    # data = pd.read_parquet(f'data/input/city_wise_train/{city}_data_train.parquet')
    processed_data , _ = data_preprocess(data, feature_col = feature_col , target_col = target_col)
    processed_data, valid_cps, removed_cps = make_others(processed_data, target_col, min_samples = 150, max_samples=None, valid_cps=None)

    encoder = LabelEncoder()
    processed_data[target_col] = encoder.fit_transform(processed_data[target_col])


    # ============================
    # 3. Define splits
    # ============================

    train_df, val_df = train_test_split(
        processed_data,
        test_size=0.05,
        random_state=42,
        stratify=processed_data[target_col]
    )

    # ============================
    # 4. Define vectorizer & model
    # ============================

    vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(2, 3), 
        max_features=5000)

    model = LogisticRegression(
        max_iter=100, 
        multi_class='multinomial', 
        solver='lbfgs', 
        random_state=42)

    # ============================
    # 5. Initialize classifier
    # ============================
    clf = AddressClassifier(
        train_data=train_df,
        val_data=val_df,
        train_feature=feature_col,
        train_target=target_col,
        encoder=encoder,
        model=model,
        vectorizer=vectorizer,
        directory=MODEL_DIRECTORY
    )

    # ============================
    # 6. Train the model
    # ============================
    clf.train_model()


    # ============================
    # 7. Save model & vectorizer
    # ============================
    clf.save_model()

    processed_data,_ = data_preprocess(test_data, feature_col = feature_col , target_col = target_col)
    print(f'processed_data_test_data{processed_data.shape}')

    predata_test_data, valid_cps , removed_cps_test = make_others(processed_data, target_col, valid_cps=valid_cps)
    address_inference = AddressClassifierInference(MODEL_DIRECTORY)
    address_test_df = address_inference.predict(data=predata_test_data, feature_col = 'full_address', target_col = 'cp_code')

    pd.DataFrame(removed_cps).to_csv(address_inference.other_cp_path, index=False)
    pd.DataFrame(valid_cps).to_csv(address_inference.trained_cp_path, index=False)
    address_test_df.to_parquet(address_inference.test_accuracy_data_path, index=False)
    

if __name__=="__main__":
    # pincode = list(range(59,65)) + list(range(67,85))
    pincode = ['68']
    print(pincode)

    def run_city_model(pincode):
        data = pd.read_parquet(f'/home/ds/cp_prediction_pincode/data/instance_train/{pincode}_data_train.parquet')
        test_data = pd.read_parquet(f'/home/ds/cp_prediction_pincode/data/instance_test/{pincode}_data_test.parquet')
        city_models(pincode, data, test_data)


    for p in pincode:
        print(f'running for {p}')
        run_city_model(p)
    # with mp.Pool(processes=4) as pool:
    #     pool.map(run_city_model, pincode)


