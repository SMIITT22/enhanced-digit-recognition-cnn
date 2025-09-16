import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageTk, ImageFont
import cv2
import threading
import os
import requests
import gzip
import pickle
import io
from sklearn.model_selection import train_test_split

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Digit Recognition CNN (0-10)")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.is_training = False
        self.canvas_size = 280
        self.drawing = False
        
        # Create main frames
        self.create_widgets()
        
        # Create initial model architecture
        self.create_model()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel for drawing and controls
        left_frame = ttk.LabelFrame(main_frame, text="Drawing & Controls", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Drawing canvas
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg='black', cursor='pencil')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Bind mouse events for drawing
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Control buttons
        ttk.Button(left_frame, text="Clear Canvas", command=self.clear_canvas).grid(
            row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(left_frame, text="Predict Digit", command=self.predict_digit).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Prediction result
        self.prediction_label = ttk.Label(left_frame, text="Draw a digit and click 'Predict'", 
                                        font=('Arial', 12), justify='center')
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Training controls
        training_frame = ttk.LabelFrame(left_frame, text="Training Configuration", padding="10")
        training_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Training parameters
        ttk.Label(training_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="10")
        ttk.Entry(training_frame, textvariable=self.epochs_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(training_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(training_frame, textvariable=self.batch_size_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Dataset info
        self.dataset_info_label = ttk.Label(training_frame, 
                                          text="Dataset: MNIST + EMNIST + Synthetic 10s", 
                                          font=('Arial', 9))
        self.dataset_info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Training button
        self.train_button = ttk.Button(training_frame, text="Train Enhanced Model", 
                                     command=self.start_training)
        self.train_button.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Progress bar
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Model management
        model_frame = ttk.LabelFrame(left_frame, text="Model Management", padding="10")
        model_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Right panel for training visualization
        right_frame = ttk.LabelFrame(main_frame, text="Training Progress & Visualization", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure for training plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Initialize empty plots
        self.ax1.set_title('Model Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Model Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.grid(True, alpha=0.3)
        
        # Create canvas for matplotlib
        self.plot_canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to train or predict", 
                                    font=('Arial', 10))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
    def create_model(self):
        """Create an improved CNN model architecture"""
        self.model = models.Sequential([
            # Input layer
            layers.Input(shape=(28, 28, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block  
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(11, activation='softmax')  # 11 classes (0-10)
        ])
        
        # Use advanced optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary for debugging
        print("Model Architecture:")
        self.model.summary()
        
        self.status_label.config(text="Enhanced CNN model created successfully")
        
    def download_emnist_dataset(self):
        """Download and load EMNIST digits dataset"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Attempting to download EMNIST dataset..."))
            
            # EMNIST digits dataset URLs
            base_url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
            files = {
                'train_images': 'emnist-digits-train-images-idx3-ubyte.gz',
                'train_labels': 'emnist-digits-train-labels-idx1-ubyte.gz',
                'test_images': 'emnist-digits-test-images-idx3-ubyte.gz',
                'test_labels': 'emnist-digits-test-labels-idx1-ubyte.gz'
            }
            
            # Create data directory
            os.makedirs('emnist_data', exist_ok=True)
            
            def download_file(url, filename):
                if not os.path.exists(f'emnist_data/{filename}'):
                    print(f"Downloading {filename}...")
                    response = requests.get(url + filename, stream=True, timeout=30)
                    response.raise_for_status()
                    with open(f'emnist_data/{filename}', 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded {filename}")
            
            # Download files
            for key, filename in files.items():
                download_file(base_url, filename)
            
            # Load EMNIST data
            def load_emnist_images(filename):
                with gzip.open(f'emnist_data/{filename}', 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                    data = data.reshape(-1, 28, 28)
                    # EMNIST images need to be rotated and flipped
                    data = np.rot90(data, k=3, axes=(1, 2))
                    data = np.fliplr(data)
                    return data
            
            def load_emnist_labels(filename):
                with gzip.open(f'emnist_data/{filename}', 'rb') as f:
                    return np.frombuffer(f.read(), np.uint8, offset=8)
            
            train_images = load_emnist_images('emnist-digits-train-images-idx3-ubyte.gz')
            train_labels = load_emnist_labels('emnist-digits-train-labels-idx1-ubyte.gz')
            test_images = load_emnist_images('emnist-digits-test-images-idx3-ubyte.gz')
            test_labels = load_emnist_labels('emnist-digits-test-labels-idx1-ubyte.gz')
            
            print(f"EMNIST loaded: {len(train_images)} train, {len(test_images)} test samples")
            return (train_images, train_labels), (test_images, test_labels)
            
        except Exception as e:
            print(f"Could not download EMNIST: {e}")
            self.root.after(0, lambda: self.status_label.config(text="EMNIST download failed, using MNIST only"))
            return None, None
    
    def create_synthetic_digit_10(self, num_samples):
        """Create diverse synthetic digit 10 samples"""
        synthetic_images = []
        
        for i in range(num_samples):
            # Create blank image
            img = np.zeros((28, 28), dtype=np.float32)
            
            # Random positioning and styling
            style = np.random.choice(['side_by_side', 'overlapped', 'stacked'])
            
            if style == 'side_by_side':
                # Draw "1" on the left
                start_col = np.random.randint(2, 6)
                thickness = np.random.randint(1, 3)
                height_start = np.random.randint(4, 8)
                height_end = np.random.randint(20, 24)
                
                # Vertical line for "1"
                img[height_start:height_end, start_col:start_col+thickness] = 1.0
                # Top diagonal for "1"
                if np.random.random() > 0.5:
                    for j in range(3):
                        if start_col-j >= 0 and height_start+j < 28:
                            img[height_start+j, start_col-j] = 1.0
                
                # Draw "0" on the right
                center_col = np.random.randint(18, 22)
                center_row = np.random.randint(12, 16)
                width = np.random.randint(4, 7)
                height = np.random.randint(6, 10)
                thickness_0 = np.random.randint(1, 3)
                
                # Create oval shape for "0"
                y, x = np.ogrid[:28, :28]
                mask = ((x - center_col)**2 / (width/2)**2 + (y - center_row)**2 / (height/2)**2) <= 1
                inner_mask = ((x - center_col)**2 / ((width/2)-thickness_0)**2 + (y - center_row)**2 / ((height/2)-thickness_0)**2) <= 1
                img[mask] = 1.0
                img[inner_mask] = 0.0
                
            elif style == 'overlapped':
                # Overlapping "10"
                # Draw "1"
                start_col = np.random.randint(8, 12)
                img[6:22, start_col:start_col+2] = 1.0
                
                # Draw "0" partially overlapping
                center_col = np.random.randint(14, 18)
                center_row = 14
                cv2.ellipse(img, (center_col, center_row), (4, 8), 0, 0, 360, 1.0, 2)
                cv2.ellipse(img, (center_col, center_row), (2, 6), 0, 0, 360, 0.0, -1)
                
            else:  # stacked
                # Stack "1" on top of "0"
                # Draw "1" in upper half
                start_col = np.random.randint(12, 16)
                img[4:14, start_col:start_col+2] = 1.0
                
                # Draw "0" in lower half
                center_col = np.random.randint(12, 16)
                center_row = 20
                cv2.ellipse(img, (center_col, center_row), (5, 4), 0, 0, 360, 1.0, 2)
                cv2.ellipse(img, (center_col, center_row), (3, 2), 0, 0, 360, 0.0, -1)
            
            # Add random transformations
            if np.random.random() > 0.5:
                # Random rotation (-15 to 15 degrees)
                angle = np.random.uniform(-15, 15)
                center = (14, 14)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (28, 28))
            
            # Add noise
            noise = np.random.normal(0, 0.05, (28, 28))
            img = np.clip(img + noise, 0, 1)
            
            # Random scaling
            if np.random.random() > 0.7:
                scale = np.random.uniform(0.8, 1.2)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 0 and new_w > 0:
                    resized = cv2.resize(img, (new_w, new_h))
                    img = np.zeros((28, 28))
                    # Center the resized image
                    start_y = max(0, (28 - new_h) // 2)
                    start_x = max(0, (28 - new_w) // 2)
                    end_y = min(28, start_y + new_h)
                    end_x = min(28, start_x + new_w)
                    img[start_y:end_y, start_x:end_x] = resized[:end_y-start_y, :end_x-start_x]
            
            synthetic_images.append(img)
        
        return np.array(synthetic_images)

    def prepare_comprehensive_dataset(self):
        """Prepare comprehensive dataset from multiple sources"""
        self.root.after(0, lambda: self.status_label.config(text="Loading MNIST dataset..."))
        
        # Load MNIST data
        (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
        
        # Normalize MNIST
        mnist_x_train = mnist_x_train.astype('float32') / 255.0
        mnist_x_test = mnist_x_test.astype('float32') / 255.0
        
        all_train_x = [mnist_x_train]
        all_train_y = [mnist_y_train]
        all_test_x = [mnist_x_test]
        all_test_y = [mnist_y_test]
        
        # Try to load EMNIST
        self.root.after(0, lambda: self.status_label.config(text="Attempting to load EMNIST dataset..."))
        emnist_train, emnist_test = self.download_emnist_dataset()
        
        if emnist_train is not None:
            emnist_x_train, emnist_y_train = emnist_train
            emnist_x_test, emnist_y_test = emnist_test
            
            # Normalize EMNIST
            emnist_x_train = emnist_x_train.astype('float32') / 255.0
            emnist_x_test = emnist_x_test.astype('float32') / 255.0
            
            all_train_x.append(emnist_x_train)
            all_train_y.append(emnist_y_train)
            all_test_x.append(emnist_x_test)
            all_test_y.append(emnist_y_test)
            
            self.root.after(0, lambda: self.status_label.config(text="EMNIST loaded successfully"))
        else:
            self.root.after(0, lambda: self.status_label.config(text="Using MNIST only (EMNIST unavailable)"))
        
        # Create synthetic digit 10 data
        self.root.after(0, lambda: self.status_label.config(text="Creating synthetic digit 10 data..."))
        
        total_samples = sum(len(x) for x in all_train_x)
        synthetic_train_count = min(20000, total_samples // 10)  # 10% of total data or 20k max
        synthetic_test_count = synthetic_train_count // 4  # 25% for testing
        
        synthetic_train_10 = self.create_synthetic_digit_10(synthetic_train_count)
        synthetic_test_10 = self.create_synthetic_digit_10(synthetic_test_count)
        
        # Combine all training data
        x_train = np.concatenate(all_train_x + [synthetic_train_10])
        y_train = np.concatenate(all_train_y + [np.full(len(synthetic_train_10), 10)])
        
        # Combine all test data
        x_test = np.concatenate(all_test_x + [synthetic_test_10])
        y_test = np.concatenate(all_test_y + [np.full(len(synthetic_test_10), 10)])
        
        # Add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Shuffle the data
        train_indices = np.random.permutation(len(x_train))
        test_indices = np.random.permutation(len(x_test))
        
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, 11)
        y_test = to_categorical(y_test, 11)
        
        self.root.after(0, lambda: self.status_label.config(
            text=f"Dataset prepared: {len(x_train)} training, {len(x_test)} test samples"))
        
        return (x_train, y_train), (x_test, y_test)
        
    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress!")
            return
            
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size")
            return
            
        # Start training in separate thread to prevent GUI freezing
        self.is_training = True
        self.train_button.config(state='disabled')
        self.progress.start()
        
        training_thread = threading.Thread(target=self.train_model, args=(epochs, batch_size))
        training_thread.daemon = True
        training_thread.start()
        
    def train_model(self, epochs, batch_size):
        """Train the model with comprehensive dataset and data augmentation"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Preparing comprehensive dataset..."))
            
            # Prepare the comprehensive dataset
            (x_train, y_train), (x_test, y_test) = self.prepare_comprehensive_dataset()
            
            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                fill_mode='constant',
                cval=0
            )
            
            datagen.fit(x_train)
            
            self.root.after(0, lambda: self.status_label.config(text="Training enhanced model..."))
            
            # Custom callback for updating plots
            class EnhancedCallback(tf.keras.callbacks.Callback):
                def __init__(self, app):
                    self.app = app
                    self.losses = []
                    self.accuracies = []
                    self.val_losses = []
                    self.val_accuracies = []
                    
                def on_epoch_end(self, epoch, logs=None):
                    self.losses.append(logs['loss'])
                    self.accuracies.append(logs['accuracy'])
                    self.val_losses.append(logs['val_loss'])
                    self.val_accuracies.append(logs['val_accuracy'])
                    
                    # Update plots on main thread
                    self.app.root.after(0, lambda: self.app.update_plots(
                        self.losses, self.accuracies, self.val_losses, self.val_accuracies))
                    
                    # Update status
                    self.app.root.after(0, lambda: self.app.status_label.config(
                        text=f"Epoch {epoch+1}/{epochs} - Acc: {logs['accuracy']:.4f} - Val Acc: {logs['val_accuracy']:.4f}"))
            
            # Callbacks for better training
            callbacks = [
                EnhancedCallback(self),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
            ]
            
            # Train the model with data augmentation
            history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Final evaluation
            final_loss, final_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            
            # Training completed
            self.root.after(0, lambda: self.training_completed_enhanced(final_accuracy))
            
        except Exception as e:
            self.root.after(0, lambda: self.training_error(str(e)))
            
    def training_completed_enhanced(self, final_accuracy):
        """Called when enhanced training is completed"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text=f"Training completed! Final accuracy: {final_accuracy:.4f}")
        messagebox.showinfo("Training Complete", 
                          f"Enhanced model training completed successfully!\n\n"
                          f"Final test accuracy: {final_accuracy:.2%}\n"
                          f"Model is ready for digit recognition (0-10).\n\n"
                          f"Try drawing digits on the canvas!")
        
    def training_error(self, error_msg):
        """Called when training encounters an error"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text=f"Training error: {error_msg}")
        messagebox.showerror("Training Error", f"Error during training:\n{error_msg}")
        
    def update_plots(self, losses, accuracies, val_losses, val_accuracies):
        """Update the training progress plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        epochs = range(1, len(losses) + 1)
        
        # Plot loss
        self.ax1.plot(epochs, losses, 'bo-', label='Training Loss', linewidth=2)
        self.ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2)
        self.ax1.set_title('Model Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        self.ax2.plot(epochs, accuracies, 'bo-', label='Training Accuracy', linewidth=2)
        self.ax2.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy', linewidth=2)
        self.ax2.set_title('Model Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.plot_canvas.draw()
        
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.draw(event)
        
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing:
            x, y = event.x, event.y
            # Draw a circle (brush)
            r = 8  # brush radius
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
            
    def stop_drawing(self, event):
        """Stop drawing"""
        self.drawing = False
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a digit and click 'Predict'")
        
    def get_canvas_image(self):
        """Convert canvas drawing to 28x28 numpy array"""
        # Create image from canvas items
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'black')
        draw = ImageDraw.Draw(img)
        
        # Get all canvas items and recreate them
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # oval
                draw.ellipse(coords, fill='white')
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply preprocessing similar to training data
        # Center the image
        center_of_mass = self.calculate_center_of_mass(img_array)
        img_array = self.center_image(img_array, center_of_mass)
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def calculate_center_of_mass(self, img):
        """Calculate center of mass of the image"""
        h, w = img.shape
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        total_mass = np.sum(img)
        if total_mass == 0:
            return (w//2, h//2)
        
        center_x = np.sum(x_coords * img) / total_mass
        center_y = np.sum(y_coords * img) / total_mass
        
        return (center_x, center_y)
    
    def center_image(self, img, center_of_mass):
        """Center the image based on center of mass"""
        h, w = img.shape
        center_x, center_y = center_of_mass
        
        # Calculate shift needed to center
        shift_x = w // 2 - center_x
        shift_y = h // 2 - center_y
        
        # Apply translation
        M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
        centered_img = cv2.warpAffine(img, M, (w, h))
        
        return centered_img
        
    def predict_digit(self):
        """Predict the digit from canvas drawing with confidence analysis"""
        if self.model is None:
            messagebox.showerror("Error", "No model available. Please train the model first.")
            return
            
        try:
            # Get image from canvas
            img_array = self.get_canvas_image()
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_indices:
                top_3_predictions.append(f"{idx}: {predictions[0][idx]*100:.1f}%")
            
            # Update prediction label with enhanced info
            prediction_text = f"Predicted Digit: {predicted_digit}\n"
            prediction_text += f"Confidence: {confidence:.1f}%\n"
            prediction_text += f"Top 3: {', '.join(top_3_predictions)}"
            
            self.prediction_label.config(text=prediction_text)
            
            # Update status with color coding based on confidence
            if confidence > 90:
                status_text = f"High confidence prediction: {predicted_digit} ({confidence:.1f}%)"
            elif confidence > 70:
                status_text = f"Medium confidence prediction: {predicted_digit} ({confidence:.1f}%)"
            else:
                status_text = f"Low confidence prediction: {predicted_digit} ({confidence:.1f}%)"
                
            self.status_label.config(text=status_text)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction:\n{str(e)}")
            
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            messagebox.showerror("Error", "No model to save. Please train the model first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".keras",
            filetypes=[("Keras models", "*.keras"), ("H5 models", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.model.save(file_path)
                messagebox.showinfo("Success", f"Model saved successfully to:\n{file_path}")
                self.status_label.config(text=f"Model saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")
                
    def load_model(self):
        """Load a trained model"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Keras models", "*.keras"), ("H5 models", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.model = tf.keras.models.load_model(file_path)
                messagebox.showinfo("Success", f"Model loaded successfully from:\n{file_path}")
                self.status_label.config(text=f"Model loaded from {os.path.basename(file_path)}")
                
                # Update prediction label
                self.prediction_label.config(text="Model loaded! Draw a digit and click 'Predict'")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

def main():
    """Main function to run the enhanced digit recognition application"""
    print("=" * 60)
    print("ENHANCED DIGIT RECOGNITION CNN (0-10)")
    print("=" * 60)
    print("Features:")
    print("• MNIST Dataset (70k samples)")
    print("• EMNIST Dataset (280k additional samples)")
    print("• Synthetic Digit 10 Generation")
    print("• Advanced CNN Architecture")
    print("• Data Augmentation")
    print("• Real-time Training Visualization")
    print("• Confidence Analysis")
    print("• Model Save/Load Functionality")
    print()
    print("Required packages:")
    print("pip install tensorflow matplotlib pillow opencv-python scikit-learn requests")
    print()
    print("Instructions:")
    print("1. Click 'Train Enhanced Model' to start training")
    print("2. Wait for training to complete (10-20 minutes)")
    print("3. Draw digits 0-10 on the canvas")
    print("4. Click 'Predict Digit' to see results")
    print("5. Save your trained model for future use")
    print("=" * 60)
    
    try:
        # Set up matplotlib backend for better compatibility
        plt.style.use('default')
        
        root = tk.Tk()
        
        # Set icon and additional window properties
        root.resizable(True, True)
        root.minsize(1000, 700)
        
        # Center the window on screen
        root.update_idletasks()
        width = 1200
        height = 800
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create application
        app = DigitRecognitionApp(root)
        
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Application Error", f"An error occurred:\n{str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()