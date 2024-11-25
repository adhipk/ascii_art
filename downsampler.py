import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange
from numba.types import float32, int32
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional


@njit(parallel=True, fastmath=True, cache=True)
def fast_patch_processing(patches_original, patches_gaussian, gaussian_kernel, beta):
    """Optimized patch processing using Numba with type hints."""
    h, w, kh, kw = patches_original.shape
    result = np.empty((h, w), dtype=np.float32)
    
    for i in prange(h):
        for j in range(w):
            patch_orig = patches_original[i, j]
            patch_gauss = patches_gaussian[i, j]
            
            diff = np.abs(patch_orig - patch_gauss)
            max_diff = max(np.max(diff), 1e-6)
            dx = diff / (beta * max_diff)
            
            fx = dx * gaussian_kernel
            fx_sum = np.sum(fx) + 1e-10
            fx = fx / fx_sum
            
            result[i, j] = np.sum(patch_orig * fx)
    
    return result

@njit(parallel=True, fastmath=True, cache=True)
def create_gaussian_kernel(size, sigma):
    """Fast Gaussian kernel creation with Numba."""
    center = size // 2
    x = np.arange(size, dtype=np.float32) - center
    g = np.exp(-x**2 / (2 * sigma**2))
    return np.outer(g, g) / np.sum(g)**2

class FeaturePreservingDownsampler:
    def __init__(self, radius: int = 2, beta: float = 1.0):
        self.radius = radius
        self.beta = beta
        self._cached_gaussian_kernel: Optional[np.ndarray] = None
        self._cached_kernel_size: Optional[int] = None
        self._cached_scale_factor: Optional[int] = None
        self._preallocated_buffers: Dict[Tuple[int, int], np.ndarray] = {}
        
    def _get_or_create_buffer(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get or create preallocated buffer for given shape."""
        if shape not in self._preallocated_buffers:
            self._preallocated_buffers[shape] = np.empty(shape, dtype=np.float32)
        return self._preallocated_buffers[shape]
        
    def create_or_get_gaussian_kernel(self, size: int) -> np.ndarray:
        """Cached kernel getter with size validation."""
        if size < 3 or size % 2 == 0:
            raise ValueError("Kernel size must be odd and >= 3")
            
        if size == self._cached_kernel_size and self._cached_gaussian_kernel is not None:
            return self._cached_gaussian_kernel
            
        sigma = size / 3.0
        kernel = create_gaussian_kernel(size, sigma)
        self._cached_gaussian_kernel = kernel
        self._cached_kernel_size = size
        return kernel
    
    def extract_patches_strided(self, padded_image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Memory-efficient patch extraction using memory views."""
        view = memoryview(padded_image)
        shape = (
            padded_image.shape[0] - kernel_size + 1,
            padded_image.shape[1] - kernel_size + 1,
            kernel_size, kernel_size
        )
        strides = (
            padded_image.strides[0],
            padded_image.strides[1],
            padded_image.strides[0],
            padded_image.strides[1]
        )
        return as_strided(padded_image, shape=shape, strides=strides, writeable=False)
    
    def downsample(self, image: np.ndarray, k: int) -> np.ndarray:
        """Optimized downsampling with input validation and caching."""
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input must be 2D numpy array")
        if k <= 0:
            return image
            
        image = image.astype(np.float32, copy=False)
        scale_factor = 2**k
        self._cached_scale_factor = scale_factor
        
        h, w = image.shape
        new_h, new_w = h//scale_factor, w//scale_factor
        
        # Optimized gaussian downsampling
        sigma = scale_factor/2
        gaussian_buffer = self._get_or_create_buffer((h, w))
        cv2.GaussianBlur(image, (0, 0), sigma, dst=gaussian_buffer)
        
        gaussian_downsampled = cv2.resize(
            gaussian_buffer,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        gaussian_upsampled = cv2.resize(
            gaussian_downsampled,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Kernel and padding preparation
        kernel_size = 2 * self.radius + 1
        gaussian_kernel = self.create_or_get_gaussian_kernel(kernel_size)
        
        padded_image = cv2.copyMakeBorder(
            image, self.radius, self.radius, self.radius, self.radius,
            cv2.BORDER_REFLECT
        )
        padded_gaussian = cv2.copyMakeBorder(
            gaussian_upsampled, self.radius, self.radius, self.radius, self.radius,
            cv2.BORDER_REFLECT
        )
        
        # Extract and process patches
        patches_original = self.extract_patches_strided(padded_image, kernel_size)
        patches_gaussian = self.extract_patches_strided(padded_gaussian, kernel_size)
        
        filtered = fast_patch_processing(
            patches_original,
            patches_gaussian,
            gaussian_kernel,
            self.beta
        )
        
        return cv2.resize(
            filtered,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )
    
    def batch_process(self, images: List[np.ndarray], k: int) -> List[np.ndarray]:
        """Parallel batch processing of multiple images."""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda img: self.downsample(img, k), images))

def compare_downsampling_methods(image: np.ndarray, k: int = 1) -> Dict[str, np.ndarray]:
    """Optimized comparison with input validation."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be 2D numpy array")
    if k < 1:
        raise ValueError("k must be >= 1")
    
    image = image.astype(np.float32, copy=False)
    scale_factor = 2**k
    new_size = (image.shape[1]//scale_factor, image.shape[0]//scale_factor)
    
    # Pre-compute gaussian filtered
    gaussian_buffer = np.empty_like(image)
    cv2.GaussianBlur(image, (0, 0), scale_factor/2, dst=gaussian_buffer)
    
    results = {
        'nearest': cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST),
        'bilinear': cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR),
        'bicubic': cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC),
        'gaussian': cv2.resize(gaussian_buffer, new_size, interpolation=cv2.INTER_LINEAR)
    }
    
    downsampler = FeaturePreservingDownsampler(radius=2, beta=1.0)
    results['feature_preserving'] = downsampler.downsample(image, k)
    
    return results