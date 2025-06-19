import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

class GaussianVisualizer:
    def __init__(self):
        self.gif_frames = []
        
    def gaussian(self, x: np.ndarray, x0: float, bandwidth: float, y: float) -> np.ndarray:
        """Calculate gaussian function values.
        
        Args:
            x: Input x values
            x0: Mean of gaussian
            bandwidth: Standard deviation of gaussian
            y: Scale factor
            
        Returns:
            Gaussian function values
        """
        return y * np.exp(-((x - x0)**2)/(2 * bandwidth**2))
    
    def plot_gaussian_of_population(self, population: List, min_scalar: int = 0, max_scalar: int = 255, num_points: int = 1000) -> None:
        """Plot gaussian distributions for a population.
        
        Args:
            population: List of individuals with gaussian parameters
            min_scalar: Minimum x value
            max_scalar: Maximum x value 
            num_points: Number of points to plot
        """
        x_values = np.linspace(min_scalar, max_scalar, num_points)
        plt.figure(figsize=(10, 6))
        
        for individual in population:
            # For each individual, plot each Gaussian with its corresponding color
            for i in range(len(individual.gaussians)):
                x0 = individual.gaussians[i].opacity[0]
                bandwidth = individual.gaussians[i].opacity[1]
                y = individual.gaussians[i].opacity[2]
                color_idx = i
                if color_idx < len(individual.color):
                    # Get RGB color values, normalize to [0,1] if needed
                    r, g, b = individual.gaussians[color_idx].color
                    if r > 1.0 or g > 1.0 or b > 1.0:
                        r, g, b = r/255.0, g/255.0, b/255.0
                    color = (r, g, b)
                else:
                    # Fallback color if for some reason the indices don't match
                    color = 'blue'
                
                # Calculate and plot the Gaussian curve with its corresponding color
                gaussian_curve = y * np.exp(-((x_values - x0)**2) / (2 * bandwidth**2))
                plt.plot(x_values, gaussian_curve, alpha=0.6, linewidth=1.2, color=color)
        
        plt.title(f'Gaussian Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # Save plot to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frame = Image.open(buf)
        self.gif_frames.append(frame)

        plt.close()
        
    def gen_gif_from_images(self, output_gif_path: str = "graph.gif", duration: int = 500, loop: int = 2) -> None:
        """Generate GIF from stored frames.
        
        Args:
            output_gif_path: Path to save the output GIF
            duration: Duration for each frame in milliseconds
            loop: Number of times to loop the GIF
        """
        if not self.gif_frames:
            print("No frames to generate GIF")
            return
            
        print(f"Number of frames: {len(self.gif_frames)}")
        self.gif_frames[0].save(
            output_gif_path,
            format='GIF',
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF saved to {output_gif_path}")
        
    def clear_frames(self) -> None:
        """Clear stored animation frames."""
        self.gif_frames = [] 