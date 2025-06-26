import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import entropy
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel, laplace
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class NoReferenceImageQualityAnalyzer:
    """
    No-reference (blind) image quality analyzer for comparing super-resolution methods
    without requiring ground truth images
    """
    
    def __init__(self):
        self.methods = ['Bicubic', 'SRCNN', 'ESRGAN']
        self.metrics = {}
    
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Load and preprocess images"""
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                print(f"Warning: Could not load image {path}")
        return images
    
    def calculate_sharpness_laplacian(self, img: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return laplacian.var()
    
    def calculate_sharpness_brenner(self, img: np.ndarray) -> float:
        """Calculate Brenner sharpness measure"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Brenner measure
        brenner = np.sum((img_gray[:-2, :] - img_gray[2:, :]) ** 2)
        return brenner / (img_gray.shape[0] * img_gray.shape[1])
    
    def calculate_gradient_magnitude(self, img: np.ndarray) -> float:
        """Calculate average gradient magnitude"""
        if len(img.shape) == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img
        
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(grad_magnitude)
    
    def calculate_edge_density(self, img: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return edge_density
    
    def calculate_contrast_michelson(self, img: np.ndarray) -> float:
        """Calculate Michelson contrast"""
        if len(img.shape) == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img
        
        img_max = np.max(img_gray)
        img_min = np.min(img_gray)
        
        if img_max + img_min > 0:
            contrast = (img_max - img_min) / (img_max + img_min)
        else:
            contrast = 0
        return contrast
    
    def calculate_contrast_rms(self, img: np.ndarray) -> float:
        """Calculate RMS contrast"""
        if len(img.shape) == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img
        
        mean_val = np.mean(img_gray)
        rms_contrast = np.sqrt(np.mean((img_gray - mean_val) ** 2))
        return rms_contrast
    
    def calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Calculate histogram
        hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
        # Remove zero probabilities
        hist = hist[hist > 0]
        # Calculate entropy
        return -np.sum(hist * np.log2(hist))
    
    def calculate_local_contrast(self, img: np.ndarray) -> float:
        """Calculate local contrast using standard deviation"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Apply local standard deviation filter
        kernel = np.ones((9, 9), np.float32) / 81
        mean_img = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
        sqr_img = cv2.filter2D((img_gray.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(sqr_img - mean_img**2)
        
        return np.mean(local_std)
    
    def calculate_texture_strength(self, img: np.ndarray) -> float:
        """Calculate texture strength using Local Binary Patterns"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Calculate LBP
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        
        # Texture strength as histogram entropy
        hist = hist[hist > 0]
        texture_strength = -np.sum(hist * np.log2(hist))
        
        return texture_strength
    
    def calculate_colorfulness(self, img: np.ndarray) -> float:
        """Calculate colorfulness metric"""
        if len(img.shape) != 3:
            return 0.0
        
        # Convert to float
        img_float = img.astype(np.float32)
        
        # Calculate rg and yb
        R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Calculate std and mean
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        
        # Calculate colorfulness
        std_rgyb = np.sqrt(std_rg**2 + std_yb**2)
        mean_rgyb = np.sqrt(mean_rg**2 + mean_yb**2)
        
        colorfulness = std_rgyb + 0.3 * mean_rgyb
        return colorfulness
    
    def calculate_noise_level(self, img: np.ndarray) -> float:
        """Estimate noise level using high-frequency content"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Apply high-pass filter to estimate noise
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        filtered = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
        noise_level = np.std(filtered)
        
        return noise_level
    
    def calculate_frequency_domain_metrics(self, img: np.ndarray) -> Dict[str, float]:
        """Calculate frequency domain metrics"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # FFT
        fft = np.fft.fft2(img_gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # High frequency energy
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define high frequency region (outer 30% of spectrum)
        mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask_radius = min(h, w) * 0.35
        mask = (x - center_w)**2 + (y - center_h)**2 > mask_radius**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask]) / np.sum(magnitude_spectrum)
        
        # Spectral entropy
        power_spectrum = magnitude_spectrum**2
        power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
        power_spectrum_norm = power_spectrum_norm[power_spectrum_norm > 0]
        spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm))
        
        return {
            'high_freq_energy': high_freq_energy,
            'spectral_entropy': spectral_entropy
        }
    
    def calculate_brisque_features(self, img: np.ndarray) -> float:
        """Simplified BRISQUE-like features"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            img_gray = img.astype(np.float32)
        
        # Normalize
        img_gray = (img_gray - np.mean(img_gray)) / (np.std(img_gray) + 1e-10)
        
        # Calculate local variance
        kernel = np.ones((3, 3), np.float32) / 9
        mu = cv2.filter2D(img_gray, -1, kernel)
        mu_sq = cv2.filter2D(img_gray * img_gray, -1, kernel)
        sigma = np.sqrt(np.abs(mu_sq - mu * mu))
        
        # Structural information
        structdis = (img_gray - mu) / (sigma + 1)
        
        # Calculate features (simplified)
        alpha = np.mean(structdis**2)
        return alpha
    
    def evaluate_single_image(self, img: np.ndarray) -> Dict[str, float]:
        """Evaluate all metrics for a single image"""
        # Normalize image to 0-255 range
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        
        metrics = {}
        
        # Sharpness metrics
        metrics['sharpness_laplacian'] = self.calculate_sharpness_laplacian(img)
        metrics['sharpness_brenner'] = self.calculate_sharpness_brenner(img)
        metrics['gradient_magnitude'] = self.calculate_gradient_magnitude(img)
        
        # Edge and detail metrics
        metrics['edge_density'] = self.calculate_edge_density(img)
        metrics['texture_strength'] = self.calculate_texture_strength(img)
        
        # Contrast metrics
        metrics['contrast_michelson'] = self.calculate_contrast_michelson(img)
        metrics['contrast_rms'] = self.calculate_contrast_rms(img)
        metrics['local_contrast'] = self.calculate_local_contrast(img)
        
        # Information content
        metrics['entropy'] = self.calculate_entropy(img)
        metrics['colorfulness'] = self.calculate_colorfulness(img)
        
        # Quality indicators
        metrics['noise_level'] = self.calculate_noise_level(img)
        metrics['brisque_score'] = self.calculate_brisque_features(img)
        
        # Frequency domain metrics
        freq_metrics = self.calculate_frequency_domain_metrics(img)
        metrics.update(freq_metrics)
        
        return metrics
    
    def normalize_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize images to same size"""
        if not images:
            return images
        
        # Find minimum dimensions
        min_height = min(img.shape[0] for img in images)
        min_width = min(img.shape[1] for img in images)
        
        normalized_images = []
        for img in images:
            # Resize to minimum dimensions
            resized = cv2.resize(img, (min_width, min_height))
            normalized_images.append(resized)
        
        return normalized_images
    
    def evaluate_images(self, images: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate all images"""
        if len(images) != 3:
            raise ValueError("Expected exactly 3 images (Bicubic, SRCNN, ESRGAN)")
        
        # Normalize images to same size
        norm_images = self.normalize_images(images)
        
        results = {}
        for method, img in zip(self.methods, norm_images):
            results[method] = self.evaluate_single_image(img)
        
        return results
    
    def calculate_quality_scores(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall quality scores"""
        # Define weights for different metric categories
        weights = {
            'sharpness': ['sharpness_laplacian', 'sharpness_brenner', 'gradient_magnitude'],
            'details': ['edge_density', 'texture_strength', 'high_freq_energy'],
            'contrast': ['contrast_michelson', 'contrast_rms', 'local_contrast'],
            'information': ['entropy', 'spectral_entropy'],
            'color': ['colorfulness'],
            'artifacts': ['noise_level', 'brisque_score']  # Lower is better
        }
        
        quality_scores = {}
        
        for method in self.methods:
            score_components = {}
            
            # Calculate component scores
            for category, metric_list in weights.items():
                if category == 'artifacts':
                    # For artifacts, lower is better, so invert
                    values = [1.0 / (1.0 + results[method][metric]) for metric in metric_list if metric in results[method]]
                else:
                    # For quality metrics, higher is better
                    values = [results[method][metric] for metric in metric_list if metric in results[method]]
                
                if values:
                    score_components[category] = np.mean(values)
                else:
                    score_components[category] = 0.0
            
            # Calculate weighted overall score
            overall_score = (
                score_components['sharpness'] * 0.25 +
                score_components['details'] * 0.20 +
                score_components['contrast'] * 0.20 +
                score_components['information'] * 0.15 +
                score_components['color'] * 0.10 +
                score_components['artifacts'] * 0.10
            )
            
            quality_scores[method] = {
                'overall_score': overall_score,
                **score_components
            }
        
        return quality_scores
    
    def create_comparison_report(self, results: Dict[str, Dict[str, float]], 
                               quality_scores: Dict[str, float]) -> str:
        """Generate detailed comparison report"""
        report = []
        report.append("=" * 80)
        report.append("NO-REFERENCE IMAGE QUALITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append()
        
        # Overall Quality Ranking
        report.append("1. OVERALL QUALITY RANKING")
        report.append("-" * 40)
        
        # Sort methods by overall score
        sorted_methods = sorted(quality_scores.items(), 
                              key=lambda x: x[1]['overall_score'], 
                              reverse=True)
        
        for rank, (method, scores) in enumerate(sorted_methods, 1):
            report.append(f"{rank}. {method}: {scores['overall_score']:.4f}")
        
        # Detailed Metrics Comparison
        report.append("\n\n2. DETAILED METRICS COMPARISON")
        report.append("-" * 40)
        
        # Group metrics by category
        metric_categories = {
            'Sharpness Metrics': ['sharpness_laplacian', 'sharpness_brenner', 'gradient_magnitude'],
            'Detail Metrics': ['edge_density', 'texture_strength', 'high_freq_energy'],
            'Contrast Metrics': ['contrast_michelson', 'contrast_rms', 'local_contrast'],
            'Information Metrics': ['entropy', 'spectral_entropy'],
            'Color Metrics': ['colorfulness'],
            'Quality Indicators': ['noise_level', 'brisque_score']
        }
        
        for category, metric_list in metric_categories.items():
            report.append(f"\n{category}:")
            for metric in metric_list:
                if metric in results[self.methods[0]]:
                    report.append(f"  {metric.replace('_', ' ').title()}:")
                    
                    # Get values for all methods
                    values = [(method, results[method][metric]) for method in self.methods]
                    
                    # Sort by value (descending for most metrics, ascending for noise/artifacts)
                    reverse_sort = metric not in ['noise_level', 'brisque_score']
                    values.sort(key=lambda x: x[1], reverse=reverse_sort)
                    
                    for rank, (method, value) in enumerate(values, 1):
                        indicator = "↑" if reverse_sort else "↓"
                        report.append(f"    {rank}. {method}: {value:.4f} {indicator if rank == 1 else ''}")
        
        # Performance Analysis
        report.append("\n\n3. PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        # Find best performing method
        best_method = sorted_methods[0][0]
        best_score = sorted_methods[0][1]['overall_score']
        
        report.append(f"Best Performing Method: {best_method}")
        report.append(f"Best Overall Score: {best_score:.4f}")
        
        # Calculate improvements
        if len(sorted_methods) > 1:
            second_best = sorted_methods[1]
            improvement = ((best_score - second_best[1]['overall_score']) / second_best[1]['overall_score']) * 100
            report.append(f"Improvement over {second_best[0]}: {improvement:.1f}%")
        
        # Category Analysis
        report.append(f"\n{best_method} excels in:")
        best_scores = quality_scores[best_method]
        category_performance = [(cat, score) for cat, score in best_scores.items() 
                              if cat != 'overall_score']
        category_performance.sort(key=lambda x: x[1], reverse=True)
        
        for category, score in category_performance[:3]:
            report.append(f"  - {category.replace('_', ' ').title()}: {score:.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def visualize_metrics(self, results: Dict[str, Dict[str, float]], 
                         quality_scores: Dict[str, float]):
        """Create comprehensive visualizations"""
        
        # 1. Overall Quality Scores
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('No-Reference Image Quality Assessment', fontsize=16, fontweight='bold')
        
        # Overall scores
        methods = list(quality_scores.keys())
        overall_scores = [quality_scores[method]['overall_score'] for method in methods]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars1 = ax1.bar(methods, overall_scores, color=colors)
        ax1.set_title('Overall Quality Scores', fontweight='bold')
        ax1.set_ylabel('Quality Score')
        
        # Add value labels
        for bar, score in zip(bars1, overall_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Highlight best performer
        best_idx = np.argmax(overall_scores)
        bars1[best_idx].set_edgecolor('gold')
        bars1[best_idx].set_linewidth(3)
        
        # 2. Sharpness Comparison
        sharpness_metrics = ['sharpness_laplacian', 'sharpness_brenner', 'gradient_magnitude']
        sharpness_data = []
        for method in methods:
            method_data = [results[method][metric] for metric in sharpness_metrics]
            sharpness_data.append(method_data)
        
        x = np.arange(len(sharpness_metrics))
        width = 0.25
        
        for i, (method, data) in enumerate(zip(methods, sharpness_data)):
            ax2.bar(x + i*width, data, width, label=method, color=colors[i])
        
        ax2.set_title('Sharpness Metrics Comparison', fontweight='bold')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in sharpness_metrics], rotation=45)
        ax2.legend()
        
        # 3. Detail and Texture Metrics
        detail_metrics = ['edge_density', 'texture_strength', 'entropy']
        detail_data = []
        for method in methods:
            method_data = [results[method][metric] for metric in detail_metrics]
            detail_data.append(method_data)
        
        x = np.arange(len(detail_metrics))
        for i, (method, data) in enumerate(zip(methods, detail_data)):
            ax3.bar(x + i*width, data, width, label=method, color=colors[i])
        
        ax3.set_title('Detail and Texture Metrics', fontweight='bold')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in detail_metrics], rotation=45)
        ax3.legend()
        
        # 4. Quality vs Artifacts
        quality_metrics = [results[method]['contrast_rms'] for method in methods]
        artifact_metrics = [results[method]['noise_level'] for method in methods]
        
        scatter = ax4.scatter(artifact_metrics, quality_metrics, 
                            c=colors, s=200, alpha=0.7)
        
        for i, method in enumerate(methods):
            ax4.annotate(method, (artifact_metrics[i], quality_metrics[i]),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax4.set_xlabel('Noise Level (Lower is Better)')
        ax4.set_ylabel('Contrast (Higher is Better)')
        ax4.set_title('Quality vs Artifacts Trade-off', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Create radar chart
        self._create_radar_chart(quality_scores)
    
    def _create_radar_chart(self, quality_scores: Dict[str, Dict[str, float]]):
        """Create radar chart for quality categories"""
        categories = ['sharpness', 'details', 'contrast', 'information', 'color', 'artifacts']
        
        # Prepare data
        methods = list(quality_scores.keys())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle('Quality Categories Comparison (Radar Chart)', fontsize=14, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, method in enumerate(methods):
            values = [quality_scores[method][cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.title() for cat in categories])
        ax.set_ylim(0, max([max(quality_scores[method][cat] for cat in categories) 
                           for method in methods]) * 1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    """Main function to demonstrate the no-reference image quality analyzer"""
    print("No-Reference Image Quality Assessment System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NoReferenceImageQualityAnalyzer()
    
    # For demonstration, create synthetic images with different quality characteristics
    print("Creating synthetic test images for demonstration...")
    
    # Create base high-quality image
    np.random.seed(42)
    base_img = np.random.rand(256, 256, 3) * 255
    base_img = base_img.astype(np.uint8)
    
    # Add some structure to make it more realistic
    for i in range(3):
        base_img[:, :, i] = cv2.GaussianBlur(base_img[:, :, i], (5, 5), 1.0)
    
    # Create some patterns
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    pattern = np.sin(x/10) * np.cos(y/10) * 50 + 128
    base_img[:, :, 0] = np.clip(base_img[:, :, 0] + pattern, 0, 255)
    
    # Simulate Bicubic (lowest quality)
    bicubic_img = cv2.resize(base_img, (64, 64))  # Downscale
    bicubic_img = cv2.resize(bicubic_img, (256, 256), interpolation=cv2.INTER_CUBIC)  # Upscale
    bicubic_img = cv2.GaussianBlur(bicubic_img, (1, 1), 0.5)  # Add slight blur
    
    # Simulate SRCNN (medium quality)
    srcnn_img = cv2.resize(base_img, (128, 128))  # Less aggressive downscale
    srcnn_img = cv2.resize(srcnn_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    # Add slight sharpening to simulate CNN enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    srcnn_img = cv2.filter2D(srcnn_img, -1, kernel * 0.1)
    srcnn_img = np.clip(srcnn_img, 0, 255).astype(np.uint8)
    
    # Simulate ESRGAN (highest quality)
    esrgan_img = base_img.copy()
    # Enhance contrast and sharpness
    esrgan_img = cv2.convertScaleAbs(esrgan_img, alpha=1.1, beta=5)
    # Add subtle sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    esrgan_img = cv2.filter2D(esrgan_img, -1, kernel * 0.05)
    esrgan_img = np.clip(esrgan_img, 0, 255).astype(np.uint8)
    
    images = [bicubic_img, srcnn_img, esrgan_img]
    
    print("Evaluating image quality metrics...")
    
    # Evaluate images
    results = analyzer.evaluate_images(images)
    
    # Calculate quality scores
    quality_scores = analyzer.calculate_quality_scores(results)
    
    # Generate report
    report = analyzer.create_comparison_report(results, quality_scores)
    print(report)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_metrics(results, quality_scores)
    
    return results, quality_scores

if __name__ == "__main__":
    # Run the demonstration
    results, quality_scores = main()
    
    print("\nDetailed Results:")
    print("=" * 50)
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nQuality Scores:")
    print("=" * 30)
    for method, scores in quality_scores.items():
        print(f"\n{method}:")
        for category, score in scores.items():
            print(f"  {category}: {score:.4f}")