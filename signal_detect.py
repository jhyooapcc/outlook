# Climate Forecast Analysis System
# Week 1: Core Foundation

import xarray as xr
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_closing, binary_opening
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnalysisConfig:
    """Tunable parameters for the analysis system"""
    
    # Probability thresholds
    probability_thresholds: Dict[str, float] = None
    
    # Spatial analysis parameters
    spatial_params: Dict[str, Any] = None
    
    # Regional attribution parameters
    region_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.probability_thresholds is None:
            self.probability_thresholds = {
                'strong_enhanced': 70,
                'enhanced': 50,
                'tendency': 40
            }
        
        if self.spatial_params is None:
            self.spatial_params = {
                'min_signal_area': 4,  # minimum grid cells for valid signal
                'connectivity': 8,     # 4 or 8-connectivity for clustering
                'remove_small_objects_threshold': 2,
                'morphology_kernel_size': 3,
                'merge_nearby_distance': 2
            }
        
        if self.region_params is None:
            self.region_params = {
                'primary_region_threshold': 70,    # % overlap to be "primary" region
                'secondary_region_threshold': 30,  # % overlap to mention as "secondary"
                'max_regions_mentioned': 3,
                'prefer_larger_regions': True
            }

class RegionalMaskLoader:
    """Load and manage regional masks"""
    
    def __init__(self, mask_file_path: str):
        self.mask_file = mask_file_path
        self.masks = None
        self.region_names = None
        self.load_masks()
    
    def load_masks(self):
        """Load regional masks from NetCDF file"""
        try:
            ds = xr.open_dataset(self.mask_file)
            print("Loaded regional masks dataset")
            print("Variables:", list(ds.variables.keys()))
            print("Dimensions:", dict(ds.dims))
            
            # Extract mask data and region information
            if 'subregion' in ds.variables:
                self.masks = ds['subregion']
                print(f"Mask shape: {self.masks.shape}")
            
            # Extract region names if available
            if 'subregion_name' in ds.variables:
                self.region_names = ds['subregion_name'].values
                print(f"Found {len(self.region_names)} regions")
                print("Sample regions:", self.region_names[:5])
            
            self.lat = ds['lat'].values
            self.lon = ds['lon'].values
            
        except Exception as e:
            print(f"Error loading masks: {e}")
            raise
    
    def get_region_mask(self, region_id: int) -> np.ndarray:
        """Get binary mask for specific region"""
        return (self.masks.values == region_id).astype(int)
    
    def get_region_name(self, region_id: int) -> str:
        """Get region name for given ID"""
        if self.region_names is not None and region_id < len(self.region_names):
            return self.region_names[region_id].decode() if isinstance(self.region_names[region_id], bytes) else str(self.region_names[region_id])
        return f"region_{region_id}"

class ForecastDataLoader:
    """Load and preprocess forecast data"""
    
    def load_netcdf_forecast(self, file_path: str) -> xr.Dataset:
        """Load precipitation probability forecast data"""
        try:
            ds = xr.open_dataset(file_path)
            print("Loaded forecast dataset")
            print("Variables:", list(ds.variables.keys()))
            print("Dimensions:", dict(ds.dims))
            return ds
        except Exception as e:
            print(f"Error loading forecast: {e}")
            raise
    
    def validate_coordinates(self, forecast_data: xr.Dataset, mask_loader: RegionalMaskLoader) -> bool:
        """Ensure coordinate alignment between forecast and masks"""
        # This will be implemented once we see the forecast data structure
        return True

class ProbabilitySignalDetector:
    """Detect probability signals using spatial coherence principles"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def detect_signals(self, prob_data: np.ndarray, condition_data: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Main detection pipeline
        
        Args:
            prob_data: 2D array of probability values (0-100)
            condition_data: 2D array indicating below_normal/above_normal
        
        Returns:
            Dictionary with detected signals by category
        """
        signals = {
            'strong_enhanced': [],
            'enhanced': [],
            'tendency': []
        }
        
        # Process each condition (below_normal, above_normal)
        for condition_value in np.unique(condition_data):
            if condition_value == 0:  # Skip no-signal areas
                continue
                
            condition_name = 'below_normal' if condition_value < 0 else 'above_normal'
            condition_mask = (condition_data == condition_value)
            
            # Apply probability thresholds
            for category, threshold in self.config.probability_thresholds.items():
                prob_mask = (prob_data >= threshold) & condition_mask
                
                if not np.any(prob_mask):
                    continue
                
                # Find connected components
                labeled_array, num_features = label(prob_mask, 
                                                  structure=np.ones((3,3)) if self.config.spatial_params['connectivity'] == 8 else None)
                
                # Process each connected component
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled_array == region_id)
                    
                    # Apply spatial coherence logic
                    if self._is_valid_signal(region_mask, prob_data, condition_mask):
                        # Determine final classification using hierarchical logic
                        final_category = self._classify_region(region_mask, prob_data)
                        
                        signal_info = {
                            'mask': region_mask,
                            'condition': condition_name,
                            'max_probability': np.max(prob_data[region_mask]),
                            'mean_probability': np.mean(prob_data[region_mask]),
                            'area': np.sum(region_mask)
                        }
                        
                        signals[final_category].append(signal_info)
        
        return signals
    
    def _is_valid_signal(self, region_mask: np.ndarray, prob_data: np.ndarray, condition_mask: np.ndarray) -> bool:
        """Check if detected region is a valid meteorological signal"""
        # Minimum area threshold
        if np.sum(region_mask) < self.config.spatial_params['min_signal_area']:
            return False
        
        # Additional validation can be added here
        return True
    
    def _classify_region(self, region_mask: np.ndarray, prob_data: np.ndarray) -> str:
        """Apply hierarchical classification logic to determine final category"""
        region_probs = prob_data[region_mask]
        
        # Count pixels in each category
        strong_pixels = np.sum(region_probs >= self.config.probability_thresholds['strong_enhanced'])
        enhanced_pixels = np.sum(region_probs >= self.config.probability_thresholds['enhanced'])
        total_pixels = len(region_probs)
        
        # Apply spatial coherence principle
        strong_ratio = strong_pixels / total_pixels
        
        # If substantial core exists at >70%, classify entire region as strong
        if strong_ratio >= 0.3:  # 30% of region has >70% probability
            return 'strong_enhanced'
        elif enhanced_pixels / total_pixels >= 0.5:  # 50% of region has >50% probability
            return 'enhanced'
        else:
            return 'tendency'

# Test with the regional masks
def test_mask_loading():
    """Test function to examine the uploaded mask file"""
    print("Testing Regional Mask Loading...")
    print("=" * 50)
    
    try:
        mask_loader = RegionalMaskLoader('subregion_masks_pacific.nc')
        
        # Display some basic information
        print(f"Coordinate ranges:")
        print(f"  Latitude: {mask_loader.lat.min():.1f} to {mask_loader.lat.max():.1f}")
        print(f"  Longitude: {mask_loader.lon.min():.1f} to {mask_loader.lon.max():.1f}")
        
        if mask_loader.region_names is not None:
            print(f"\nAll regions ({len(mask_loader.region_names)}):")
            for i, name in enumerate(mask_loader.region_names[:10]):  # Show first 10
                region_name = name.decode() if isinstance(name, bytes) else str(name)
                print(f"  {i}: {region_name}")
            if len(mask_loader.region_names) > 10:
                print(f"  ... and {len(mask_loader.region_names) - 10} more regions")
        
        return mask_loader
        
    except Exception as e:
        print(f"Error in test: {e}")
        return None

# Run the test
if __name__ == "__main__":
    mask_loader = test_mask_loading()