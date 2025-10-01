import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TCOPredictor:
    """
    ML-Model für TCO-Vorhersagen
    """
    
    def __init__(self):
        self.model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_trained = False
        self.training_stats = {}
        
    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> np.ndarray:
        """
        Konvertiert Asset-Daten in ML-Features
        
        Args:
            df: DataFrame mit Asset-Daten
            fit_encoders: Ob neue Encoder trainiert werden sollen (nur beim Training)
        
        Returns:
            Numpy array mit ML-Features
        """
        
        print("🔧 Bereite Features vor...")
        
        # Copy to avoid modifying original
        df_work = df.copy()
        
        # Handle missing values first
        df_work = self._handle_missing_values(df_work)
        
        # Numerical features (direkt verwendbar)
        numerical_features = ['purchase_price', 'age_years', 'warranty_years', 'expected_lifetime']
        
        # Categorical features (müssen encoded werden)
        categorical_features = ['category', 'subcategory', 'manufacturer', 'location', 
                              'usage_pattern', 'criticality']
        
        # Initialize feature matrix
        feature_matrix = []
        self.feature_names = []
        
        # Add numerical features
        for feature in numerical_features:
            if feature in df_work.columns:
                values = df_work[feature].fillna(df_work[feature].median())
                feature_matrix.append(values.values.reshape(-1, 1))
                self.feature_names.append(feature)
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df_work.columns:
                if fit_encoders:
                    # Training: Create new encoder
                    encoder = LabelEncoder()
                    encoded_values = encoder.fit_transform(df_work[feature].fillna('Unknown'))
                    self.feature_encoders[feature] = encoder
                else:
                    # Prediction: Use existing encoder
                    if feature in self.feature_encoders:
                        encoder = self.feature_encoders[feature]
                        # Handle unknown categories
                        values = df_work[feature].fillna('Unknown')
                        encoded_values = []
                        
                        for value in values:
                            try:
                                encoded_values.append(encoder.transform([value])[0])
                            except ValueError:
                                # Unknown category -> use most common category
                                encoded_values.append(0)
                        
                        encoded_values = np.array(encoded_values)
                    else:
                        # Fallback: alle Werte auf 0
                        encoded_values = np.zeros(len(df_work))
                
                feature_matrix.append(encoded_values.reshape(-1, 1))
                self.feature_names.append(feature)
        
        # Derived features (Feature Engineering)
        if 'purchase_price' in df_work.columns and 'age_years' in df_work.columns:
            # Price per year of age (higher = newer expensive equipment)
            price_age_ratio = df_work['purchase_price'] / (df_work['age_years'] + 1)
            feature_matrix.append(price_age_ratio.values.reshape(-1, 1))
            self.feature_names.append('price_age_ratio')
            
            # Age category (0=new, 1=medium, 2=old)
            age_category = pd.cut(df_work['age_years'], bins=[-1, 1, 3, 100], labels=[0, 1, 2])
            feature_matrix.append(age_category.values.reshape(-1, 1))
            self.feature_names.append('age_category')
        
        # Is warranty active?
        if 'age_years' in df_work.columns and 'warranty_years' in df_work.columns:
            warranty_active = (df_work['age_years'] < df_work['warranty_years']).astype(int)
            feature_matrix.append(warranty_active.values.reshape(-1, 1))
            self.feature_names.append('warranty_active')
        
        # Combine all features
        if feature_matrix:
            X = np.hstack(feature_matrix)
        else:
            raise ValueError("Keine Features konnten erstellt werden!")
        
        print(f"   ✅ {X.shape[1]} Features erstellt für {X.shape[0]} Assets")
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent Missing Value Handling"""
        
        df_clean = df.copy()
        
        # Standard defaults for missing categorical values
        categorical_defaults = {
            'manufacturer': 'Unknown',
            'usage_pattern': 'Standard (8h/Tag)',
            'criticality': 'Mittel',
            'location': 'Andere'
        }
        
        for col, default in categorical_defaults.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default)
        
        # Numerical defaults
        numerical_defaults = {
            'warranty_years': 1.0,
            'expected_lifetime': 5.0,
            'age_years': 1.0
        }
        
        for col, default in numerical_defaults.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default)
        
        return df_clean
    
    def train(self, training_data_path: str = 'data/training_data_realistic.csv') -> Dict[str, Any]:
        """
        Trainiert das ML-Model
        
        Args:
            training_data_path: Pfad zur Training-CSV
            
        Returns:
            Dictionary mit Training-Statistiken
        """
        
        print("🚀 Starte ML-Training...")
        
        # Load training data
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"Training-Daten nicht gefunden: {training_data_path}")
        
        df = pd.read_csv(training_data_path)
        print(f"📊 Geladen: {len(df)} Training-Assets")
        
        # Remove outliers (extrem unrealistische Wartungskosten)
        df_clean = self._remove_outliers(df)
        
        # Prepare features
        X = self.prepare_features(df_clean, fit_encoders=True)
        y = df_clean['annual_maintenance'].values
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"📚 Training Set: {X_train.shape[0]} Assets")
        print(f"🧪 Test Set: {X_test.shape[0]} Assets")
        
        # Scale features (wichtig für manche Algorithmen)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Model
        print("🌳 Trainiere Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=100,          # 100 Bäume
            max_depth=15,              # Verhindert Overfitting
            min_samples_split=5,       # Mindest-Samples pro Split
            min_samples_leaf=3,        # Mindest-Samples pro Blatt
            random_state=42,
            n_jobs=-1                  # Alle CPU-Kerne nutzen
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate Model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Store training stats
        self.training_stats = {
            'timestamp': datetime.now().isoformat(),
            'n_training_assets': len(X_train),
            'n_test_assets': len(X_test),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance,
            'n_features': X.shape[1]
        }
        
        self.model_trained = True
        
        # Print results
        print(f"\n🎯 Training abgeschlossen!")
        print(f"   📊 Test MAE: €{test_mae:,.0f}")
        print(f"   📊 Test R²: {test_r2:.3f} ({test_r2*100:.1f}% Varianz erklärt)")
        print(f"   📊 Test RMSE: €{test_rmse:,.0f}")
        
        print(f"\n🔍 Top 5 wichtigste Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"   • {feature}: {importance:.3f}")
        
        return self.training_stats
    
    def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Entfernt extreme Outliers basierend auf Z-Score"""
        
        # Calculate Z-score for maintenance ratio
        df['maintenance_ratio'] = df['annual_maintenance'] / df['purchase_price']
        z_scores = np.abs((df['maintenance_ratio'] - df['maintenance_ratio'].mean()) / 
                         df['maintenance_ratio'].std())
        
        # Remove extreme outliers
        df_clean = df[z_scores < threshold].copy()
        n_removed = len(df) - len(df_clean)
        
        if n_removed > 0:
            print(f"🧹 {n_removed} extreme Outliers entfernt")
        
        return df_clean.drop('maintenance_ratio', axis=1)
    
    def predict(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vorhersage für einzelnes Asset
        
        Args:
            asset_data: Dictionary mit Asset-Eigenschaften
            
        Returns:
            Dictionary mit Vorhersage und Konfidenz
        """
        
        if not self.model_trained:
            raise ValueError("Model muss erst trainiert werden!")
        
        # Convert single asset to DataFrame
        df = pd.DataFrame([asset_data])
        
        # Add derived fields if missing
        if 'age_years' not in df.columns and 'purchase_date' in df.columns:
            purchase_date = pd.to_datetime(df['purchase_date'].iloc[0])
            df['age_years'] = (datetime.now() - purchase_date).days / 365.25
        
        # Prepare features
        X = self.prepare_features(df, fit_encoders=False)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Estimate confidence using tree predictions variance
        tree_predictions = [tree.predict(X_scaled)[0] for tree in self.model.estimators_[:10]]
        prediction_std = np.std(tree_predictions)
        prediction_mean = np.mean(tree_predictions)
        
        # Calculate confidence (inversely related to variance)
        # Lower variance = higher confidence
        relative_std = prediction_std / max(prediction_mean, 1)
        confidence = max(0.5, 1 - (relative_std * 2))  # 50% minimum confidence
        
        # Determine confidence level
        if confidence >= 0.85:
            confidence_level = "Sehr Hoch"
            confidence_color = "success"
            confidence_icon = "🟢"
        elif confidence >= 0.70:
            confidence_level = "Hoch"
            confidence_color = "success"
            confidence_icon = "🟢"
        elif confidence >= 0.60:
            confidence_level = "Mittel"
            confidence_color = "warning"
            confidence_icon = "🟡"
        else:
            confidence_level = "Niedrig"
            confidence_color = "error"
            confidence_icon = "🔴"
        
        return {
            'annual_prediction': round(max(0, prediction)),  # No negative predictions
            'confidence': round(confidence * 100),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'confidence_icon': confidence_icon,
            'range_min': round(max(0, prediction * 0.8)),
            'range_max': round(prediction * 1.2),
            'prediction_std': round(prediction_std),
            'model_type': 'Random Forest'
        }
    
    def get_similar_assets(self, asset_data: Dict[str, Any], n_similar: int = 3) -> List[Dict]:
        """Findet ähnliche Assets aus Training-Daten"""
        
        if not hasattr(self, 'training_data'):
            # Load training data for similarity search
            df = pd.read_csv('data/training_data_realistic.csv')
            self.training_data = df
        
        target_category = asset_data.get('subcategory', asset_data.get('category', ''))
        target_manufacturer = asset_data.get('manufacturer', '')
        target_price = asset_data.get('purchase_price', 0)
        
        # Filter similar assets
        similar_df = self.training_data[
            (self.training_data['subcategory'] == target_category) |
            (self.training_data['manufacturer'] == target_manufacturer)
        ].copy()
        
        if len(similar_df) < n_similar:
            # Fallback to same category
            similar_df = self.training_data[
                self.training_data['category'] == asset_data.get('category', '')
            ].copy()
        
        # Calculate price similarity
        if len(similar_df) > 0 and target_price > 0:
            similar_df['price_diff'] = abs(similar_df['purchase_price'] - target_price)
            similar_df = similar_df.nsmallest(n_similar * 2, 'price_diff')
        
        # Select random sample from most similar
        similar_assets = []
        for _, row in similar_df.head(n_similar).iterrows():
            similar_assets.append({
                'name': row['asset_name'],
                'manufacturer': row['manufacturer'],
                'model': row.get('model', 'N/A'),
                'price': row['purchase_price'],
                'maintenance': row['annual_maintenance'],
                'location': row['location'],
                'age': f"{row['age_years']:.1f} Jahre"
            })
        
        return similar_assets
    
    def save_model(self, filepath: str = 'ml/tco_model.pkl'):
        """Speichert trainiertes Model"""
        
        if not self.model_trained:
            raise ValueError("Kein trainiertes Model zum Speichern!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_encoders': self.feature_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model gespeichert: {filepath}")
    
    def load_model(self, filepath: str = 'ml/tco_model.pkl'):
        """Lädt gespeichertes Model"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model-Datei nicht gefunden: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_encoders = model_data['feature_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        self.model_trained = True
        
        print(f"📁 Model geladen: {filepath}")
        print(f"   📊 R² Score: {self.training_stats['test_r2']:.3f}")
        print(f"   📊 MAE: €{self.training_stats['test_mae']:,.0f}")

if __name__ == "__main__":
    # Demo: Train and test the model
    print("🤖 TCO Predictor - ML Training Demo\n")
    
    # Initialize predictor
    predictor = TCOPredictor()
    
    # Train model
    stats = predictor.train()
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    test_asset = {
        'category': 'IT-Equipment',
        'subcategory': 'Server',
        'manufacturer': 'Dell',
        'purchase_price': 8500,
        'age_years': 2,
        'warranty_years': 3,
        'expected_lifetime': 5,
        'location': 'Düsseldorf (HQ)',
        'usage_pattern': '24/7 Betrieb',
        'criticality': 'Hoch'
    }
    
    print(f"\n🧪 Test-Vorhersage für:")
    for key, value in test_asset.items():
        print(f"   {key}: {value}")
    
    prediction = predictor.predict(test_asset)
    print(f"\n🎯 ML-Vorhersage:")
    print(f"   💰 Jährliche Wartung: €{prediction['annual_prediction']:,}")
    print(f"   🎯 Konfidenz: {prediction['confidence']}% ({prediction['confidence_level']})")
    print(f"   📊 Bereich: €{prediction['range_min']:,} - €{prediction['range_max']:,}")
    
    print(f"\n✅ Echtes ML-Model ist bereit! 🚀")