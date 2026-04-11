import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import seaborn as sns


class MultiClassEmergencySoundModel:
    def __init__(self, input_shape, num_classes=6, model_save_path='models'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        self.model = None

        # 6 classes
        self.class_names = [
            'Non-Emergency',
            'Glass Breaking',
            'Siren',
            'Baby Crying',
            'Car Horn',
            'Explosion'
        ]

    def build_model(self):
        """Build CNN model for 6-class classification"""

        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Block 4
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            # Output layer - 6 classes
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile model"""

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
            ]
        )

        print("\n✅ Model compiled successfully!")

    def calculate_class_weights(self, y_train):
        """Calculate balanced class weights"""
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"\n📊 Class Weights (to handle remaining imbalance):")
        for i, name in enumerate(self.class_names):
            if i in class_weight_dict:
                print(f"   {name:20s}: {class_weight_dict[i]:.3f}")

        return class_weight_dict

    def create_callbacks(self):
        """Create training callbacks"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model_multiclass.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.model_save_path, 'logs_multiclass', timestamp),
                histogram_freq=1
            ),
            keras.callbacks.CSVLogger(
                os.path.join(self.model_save_path, f'training_log_{timestamp}.csv')
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """Train the model"""

        class_weights = self.calculate_class_weights(y_train)
        callbacks = self.create_callbacks()

        print(f"\n🚀 Starting training...")
        print(f"   Epochs:      {epochs}")
        print(f"   Batch size:  {batch_size}")
        print(f"   Train:       {len(X_train)} samples")
        print(f"   Validation:  {len(X_val)} samples")
        print(f"   Classes:     {self.num_classes}")

        print(f"\n📊 Training set distribution:")
        for i, name in enumerate(self.class_names):
            count = np.sum(y_train == i)
            pct = (count / len(y_train)) * 100
            print(f"   {name:20s}: {count:5d} ({pct:5.1f}%)")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""

        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)

        results = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"\n📊 Test Results:")
        print(f"   Loss:            {results[0]:.4f}")
        print(f"   Accuracy:        {results[1]*100:.2f}%")
        print(f"   Sparse Accuracy: {results[2]*100:.2f}%")
        print(f"   Top-2 Accuracy:  {results[3]*100:.2f}%")

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)

        print(f"\n📊 Confusion Matrix:")
        print(cm)

        print(f"\n📊 Per-Class Accuracy:")
        for i, name in enumerate(self.class_names):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc = np.sum((y_pred == i) & mask) / np.sum(mask)
                print(f"   {name:20s}: {class_acc*100:6.2f}%")

        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        self.plot_confusion_matrix(cm)

        print(f"\n🔍 Sample Predictions:")
        for i in range(min(15, len(X_test))):
            true_label = self.class_names[y_test[i]]
            pred_label = self.class_names[y_pred[i]]
            confidence = y_pred_proba[i][y_pred[i]] * 100
            symbol = "✅" if y_test[i] == y_pred[i] else "❌"
            print(f"   {symbol} True: {true_label:20s} | Pred: {pred_label:20s} ({confidence:5.1f}%)")

        return results, y_pred, y_pred_proba

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""

        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Percentage'})

        plt.title('Confusion Matrix - 6-Class Emergency Sound Detection',
                  fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_path = os.path.join(self.model_save_path, 'confusion_matrix_multiclass.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Confusion matrix saved: {save_path}")
        plt.close()

    def save_model(self, model_name='emergency_sound_model_multiclass.keras'):
        """Save model and class names"""
        
        model_path = os.path.join(self.model_save_path, model_name)
        self.model.save(model_path)
        print(f"\n💾 Model saved: {model_path}")

        # Save class names for Android
        class_names_path = os.path.join(self.model_save_path, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"💾 Class names saved: {class_names_path}")

        return model_path


def plot_training_history(history, save_path='models/training_history_multiclass.png'):
    """Plot training history"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sparse Accuracy
    axes[1, 0].plot(history.history['sparse_accuracy'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_sparse_accuracy'], label='Val', linewidth=2)
    axes[1, 0].set_title('Sparse Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Top-2 Accuracy
    axes[1, 1].plot(history.history['top_2_accuracy'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_top_2_accuracy'], label='Val', linewidth=2)
    axes[1, 1].set_title('Top-2 Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training history saved: {save_path}")
    plt.close()


def main():
    """Main training pipeline"""

    print("="*80)
    print("6-CLASS EMERGENCY SOUND DETECTION - MODEL TRAINING")
    print("="*80)

    # Load preprocessed data
    print("\n📂 Loading dataset...")
    data = np.load('data/processed/preprocessed_data_multiclass.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    class_names = data.get('class_names', [
        'Non-Emergency', 'Glass Breaking', 'Siren',
        'Baby Crying', 'Car Horn', 'Explosion'
    ])

    print(f"✅ Data loaded!")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test:       {len(X_test)} samples")

    input_shape = X_train.shape[1:]  # (40, 216, 1)
    num_classes = len(np.unique(y_train))

    print(f"\n📐 Input shape:  {input_shape}")
    print(f"📊 Classes:      {num_classes}")

    # Build model
    print("\n🏗️  Building model...")
    trainer = MultiClassEmergencySoundModel(input_shape, num_classes=num_classes)
    model = trainer.build_model()

    print("\n📋 Model Architecture:")
    model.summary()

    # Compile
    trainer.compile_model(learning_rate=0.001)

    # Train
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=150,
        batch_size=32
    )

    # Plot history
    plot_training_history(history)

    # Evaluate
    results, y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)

    # Save
    trainer.save_model()

    # Save training info
    training_info = {
        'model_type': '6_class_emergency_detection',
        'num_classes': num_classes,
        'class_names': trainer.class_names,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_epochs': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(results[1]),
        'test_sparse_accuracy': float(results[2]),
        'test_top2_accuracy': float(results[3]),
        'input_shape': list(input_shape),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }

    with open('models/training_info_multiclass.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Saved files:")
    print(f"   - models/best_model_multiclass.keras")
    print(f"   - models/emergency_sound_model_multiclass.keras")
    print(f"   - models/class_names.json")
    print(f"   - models/training_history_multiclass.png")
    print(f"   - models/confusion_matrix_multiclass.png")
    print(f"   - models/training_info_multiclass.json")
    print(f"\n📍 Next: python src/convert_to_tflite_multiclass.py")


if __name__ == "__main__":
    main()