#!/usr/bin/env python3
"""
Dubins Path Six Sequences Visualization Demo
Dubins路径六种序列可视化演示

Enhanced version with improved accuracy and Chinese font support

Author: AI Assistant
Date: 2025-01
Features: High-precision Dubins path calculation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
from typing import Tuple, List, Dict
import sys
import os

# Import font configuration
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    print("Warning: Could not import font_config. Chinese characters may not display correctly.")
    # Fallback font configuration
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


class HighPrecisionDubinsVisualizer:
    """High-precision Dubins path visualizer with enhanced accuracy"""
    
    def __init__(self, turning_radius: float = 2.0):
        """
        Initialize high-precision visualizer
        
        Args:
            turning_radius: Minimum turning radius
        """
        self.turning_radius = turning_radius
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
        self.colors = {
            'RSR': '#FF6B6B',  # Red
            'LSL': '#4ECDC4',  # Cyan
            'RSL': '#45B7D1',  # Blue
            'LSR': '#96CEB4',  # Green
            'RLR': '#FFEAA7',  # Yellow
            'LRL': '#DDA0DD'   # Purple
        }
        self.eps = 1e-10  # Numerical precision
        
    def mod2pi(self, angle: float) -> float:
        """Normalize angle to [0, 2π] range with high precision"""
        return angle - 2 * math.pi * math.floor(angle / (2 * math.pi))
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def coordinate_transform(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Enhanced coordinate transformation with numerical stability"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Handle zero distance case
        distance = math.sqrt(dx*dx + dy*dy)
        if distance < self.eps:
            return 0.0, 0.0, self.normalize_angle(end[2] - start[2])
        
        d = distance / self.turning_radius
        theta = math.atan2(dy, dx)
        alpha = self.normalize_angle(start[2] - theta)
        beta = self.normalize_angle(end[2] - theta)
        
        return d, alpha, beta
    
    def compute_rsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute RSR path with enhanced numerical stability"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            # Check feasibility
            denominator = d - sin_alpha + sin_beta
            if abs(denominator) < self.eps:
                return 0, 0, 0, False
            
            tmp = math.atan2(cos_alpha - cos_beta, denominator)
            
            t1 = self.mod2pi(alpha - tmp)
            
            # Enhanced path length calculation
            dx = d - sin_alpha + sin_beta
            dy = cos_alpha - cos_beta
            p_squared = dx*dx + dy*dy - 4
            
            if p_squared < -self.eps:
                return 0, 0, 0, False
            
            p = math.sqrt(max(0, p_squared))
            t2 = self.mod2pi(-beta + tmp)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_lsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute LSL path with enhanced numerical stability"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            # Check feasibility
            denominator = d + sin_alpha - sin_beta
            if abs(denominator) < self.eps:
                return 0, 0, 0, False
            
            tmp = math.atan2(cos_beta - cos_alpha, denominator)
            
            t1 = self.mod2pi(-alpha + tmp)
            
            # Enhanced path length calculation
            dx = d + sin_alpha - sin_beta
            dy = cos_beta - cos_alpha
            p_squared = dx*dx + dy*dy - 4
            
            if p_squared < -self.eps:
                return 0, 0, 0, False
            
            p = math.sqrt(max(0, p_squared))
            t2 = self.mod2pi(beta - tmp)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_rsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute RSL path with enhanced accuracy"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            p_squared = d*d - 2 + 2*math.cos(alpha - beta) - 2*d*(sin_alpha + sin_beta)
            
            if p_squared < -self.eps:
                return 0, 0, 0, False
            
            p = math.sqrt(max(0, p_squared))
            
            if p < self.eps:
                return 0, 0, 0, False
            
            tmp1 = math.atan2(cos_alpha + cos_beta, d - sin_alpha - sin_beta)
            tmp2 = math.atan2(2, p)
            tmp = tmp1 - tmp2
            
            t1 = self.mod2pi(alpha - tmp)
            t2 = self.mod2pi(beta - tmp)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_lsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute LSR path with enhanced accuracy"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            p_squared = -2 + d*d + 2*math.cos(alpha - beta) + 2*d*(sin_alpha + sin_beta)
            
            if p_squared < -self.eps:
                return 0, 0, 0, False
            
            p = math.sqrt(max(0, p_squared))
            
            if p < self.eps:
                return 0, 0, 0, False
            
            tmp1 = math.atan2(-cos_alpha - cos_beta, d + sin_alpha + sin_beta)
            tmp2 = math.atan2(-2, p)
            tmp = tmp1 - tmp2
            
            t1 = self.mod2pi(-alpha + tmp)
            t2 = self.mod2pi(-beta + tmp)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_rlr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute RLR path with enhanced accuracy"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(sin_alpha - sin_beta)) / 8
            
            if abs(tmp) > 1 + self.eps:
                return 0, 0, 0, False
            
            # Clamp to valid range
            tmp = max(-1, min(1, tmp))
            
            p = self.mod2pi(2*math.pi - math.acos(tmp))
            
            tmp1 = math.atan2(cos_alpha - cos_beta, d - sin_alpha + sin_beta)
            t1 = self.mod2pi(alpha - tmp1 + p/2)
            t2 = self.mod2pi(alpha - beta - t1 + p)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_lrl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """Compute LRL path with enhanced accuracy"""
        try:
            cos_alpha = math.cos(alpha)
            cos_beta = math.cos(beta)
            sin_alpha = math.sin(alpha)
            sin_beta = math.sin(beta)
            
            tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(sin_alpha - sin_beta)) / 8
            
            if abs(tmp) > 1 + self.eps:
                return 0, 0, 0, False
            
            # Clamp to valid range
            tmp = max(-1, min(1, tmp))
            
            p = self.mod2pi(2*math.pi - math.acos(tmp))
            
            tmp1 = math.atan2(cos_alpha - cos_beta, d - sin_alpha + sin_beta)
            t1 = self.mod2pi(-alpha + tmp1 + p/2)
            t2 = self.mod2pi(beta - alpha - t1 + p)
            
            return t1, p, t2, True
            
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0, False
    
    def compute_all_paths(self, start: Tuple[float, float, float], 
                         end: Tuple[float, float, float]) -> Dict[str, Dict]:
        """Compute all six Dubins paths with enhanced accuracy"""
        d, alpha, beta = self.coordinate_transform(start, end)
        
        paths = {}
        
        # Enhanced computation functions mapping
        compute_functions = {
            'RSR': self.compute_rsr,
            'LSL': self.compute_lsl,
            'RSL': self.compute_rsl,
            'LSR': self.compute_lsr,
            'RLR': self.compute_rlr,
            'LRL': self.compute_lrl
        }
        
        for path_type in self.path_types:
            t1, p, t2, feasible = compute_functions[path_type](d, alpha, beta)
            
            if feasible and not (math.isnan(t1) or math.isnan(p) or math.isnan(t2)):
                length = (t1 + p + t2) * self.turning_radius
            else:
                length = float('inf')
                feasible = False
            
            paths[path_type] = {
                'segments': (t1, p, t2),
                'length': length,
                'feasible': feasible,
                'normalized_params': (d, alpha, beta)
            }
        
        return paths
    
    def generate_high_precision_path_points(self, start: Tuple[float, float, float], 
                                          end: Tuple[float, float, float], 
                                          path_type: str, 
                                          num_points: int = 200) -> np.ndarray:
        """Generate high-precision path points using exact Dubins formulation"""
        paths = self.compute_all_paths(start, end)
        
        if path_type not in paths or not paths[path_type]['feasible']:
            return np.array([]).reshape(0, 2)
        
        t1, p, t2 = paths[path_type]['segments']
        
        # Use exact Dubins path generation
        return self._generate_exact_dubins_path(start, end, path_type, t1, p, t2, num_points)
    
    def _generate_exact_dubins_path(self, start: Tuple[float, float, float], 
                                   end: Tuple[float, float, float],
                                   path_type: str, t1: float, p: float, t2: float,
                                   num_points: int) -> np.ndarray:
        """Generate exact Dubins path using proper geometric construction"""
        x0, y0, theta0 = start
        
        # Calculate intermediate points using exact Dubins geometry
        points = []
        
        # Segment 1 (first turn)
        if path_type[0] == 'R':  # Right turn
            cx1 = x0 - self.turning_radius * math.sin(theta0)
            cy1 = y0 + self.turning_radius * math.cos(theta0)
            start_angle = theta0 + math.pi/2
            
            seg1_points = int(num_points * t1 / (t1 + p + t2))
            for i in range(seg1_points):
                angle = start_angle - t1 * i / max(1, seg1_points - 1)
                x = cx1 + self.turning_radius * math.cos(angle)
                y = cy1 + self.turning_radius * math.sin(angle)
                points.append([x, y])
                
        else:  # Left turn
            cx1 = x0 + self.turning_radius * math.sin(theta0)
            cy1 = y0 - self.turning_radius * math.cos(theta0)
            start_angle = theta0 - math.pi/2
            
            seg1_points = int(num_points * t1 / (t1 + p + t2))
            for i in range(seg1_points):
                angle = start_angle + t1 * i / max(1, seg1_points - 1)
                x = cx1 + self.turning_radius * math.cos(angle)
                y = cy1 + self.turning_radius * math.sin(angle)
                points.append([x, y])
        
        # Intermediate heading after first turn
        if path_type[0] == 'R':
            theta1 = theta0 - t1
        else:
            theta1 = theta0 + t1
        
        # Segment 2 (straight line)
        if len(points) > 0:
            last_point = points[-1]
        else:
            if path_type[0] == 'R':
                last_point = [cx1 + self.turning_radius * math.cos(start_angle - t1),
                             cy1 + self.turning_radius * math.sin(start_angle - t1)]
            else:
                last_point = [cx1 + self.turning_radius * math.cos(start_angle + t1),
                             cy1 + self.turning_radius * math.sin(start_angle + t1)]
        
        straight_length = p * self.turning_radius
        seg2_points = int(num_points * p / (t1 + p + t2))
        
        for i in range(seg2_points):
            t_ratio = i / max(1, seg2_points - 1)
            x = last_point[0] + straight_length * t_ratio * math.cos(theta1)
            y = last_point[1] + straight_length * t_ratio * math.sin(theta1)
            points.append([x, y])
        
        # Segment 3 (second turn)
        if len(points) > 0:
            last_point = points[-1]
        else:
            last_point = [last_point[0] + straight_length * math.cos(theta1),
                         last_point[1] + straight_length * math.sin(theta1)]
        
        if path_type[2] == 'R':  # Right turn
            cx3 = last_point[0] - self.turning_radius * math.sin(theta1)
            cy3 = last_point[1] + self.turning_radius * math.cos(theta1)
            start_angle3 = theta1 + math.pi/2
            
            seg3_points = num_points - len(points)
            for i in range(seg3_points):
                angle = start_angle3 - t2 * i / max(1, seg3_points - 1)
                x = cx3 + self.turning_radius * math.cos(angle)
                y = cy3 + self.turning_radius * math.sin(angle)
                points.append([x, y])
                
        else:  # Left turn
            cx3 = last_point[0] + self.turning_radius * math.sin(theta1)
            cy3 = last_point[1] - self.turning_radius * math.cos(theta1)
            start_angle3 = theta1 - math.pi/2
            
            seg3_points = num_points - len(points)
            for i in range(seg3_points):
                angle = start_angle3 + t2 * i / max(1, seg3_points - 1)
                x = cx3 + self.turning_radius * math.cos(angle)
                y = cy3 + self.turning_radius * math.sin(angle)
                points.append([x, y])
        
        return np.array(points) if points else np.array([]).reshape(0, 2)
    
    def plot_single_path(self, start: Tuple[float, float, float], 
                        end: Tuple[float, float, float], 
                        path_type: str, 
                        save_path: str = None):
        """Plot single path with enhanced visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate paths
        paths = self.compute_all_paths(start, end)
        
        if path_type not in paths:
            print(f"Unknown path type: {path_type}")
            return
        
        path_info = paths[path_type]
        
        # Generate high-precision path points
        if path_info['feasible']:
            path_points = self.generate_high_precision_path_points(start, end, path_type, 300)
            
            if len(path_points) > 0:
                ax.plot(path_points[:, 0], path_points[:, 1], 
                       color=self.colors[path_type], linewidth=3, 
                       label=f'{path_type} (Length: {path_info["length"]:.2f})')
                
                # Add direction arrows
                self._add_direction_arrows(ax, path_points, self.colors[path_type])
        else:
            ax.text(0.5, 0.5, f'{path_type} Path Infeasible', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, color='red')
        
        # Draw start and end poses
        self._draw_vehicle(ax, start, 'green', 'Start')
        self._draw_vehicle(ax, end, 'red', 'End')
        
        # Draw turning radius circles
        self._draw_turning_circles(ax, start, end)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_title(f'Dubins Path - {path_type}', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all_paths(self, start: Tuple[float, float, float], 
                      end: Tuple[float, float, float], 
                      save_path: str = None):
        """Plot all six path types with enhanced visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # Calculate all paths
        paths = self.compute_all_paths(start, end)
        
        for i, path_type in enumerate(self.path_types):
            ax = axes[i]
            path_info = paths[path_type]
            
            # Draw path
            if path_info['feasible']:
                path_points = self.generate_high_precision_path_points(start, end, path_type, 200)
                
                if len(path_points) > 0:
                    ax.plot(path_points[:, 0], path_points[:, 1], 
                           color=self.colors[path_type], linewidth=3)
                    
                    # Add direction arrows
                    self._add_direction_arrows(ax, path_points, self.colors[path_type])
                
                title = f'{path_type}\\nLength: {path_info["length"]:.3f}'
            else:
                title = f'{path_type}\\nInfeasible'
                ax.text(0.5, 0.5, 'Infeasible', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14, color='red')
            
            # Draw start and end poses
            self._draw_vehicle(ax, start, 'green', size=0.6)
            self._draw_vehicle(ax, end, 'red', size=0.6)
            
            # Draw turning circles
            self._draw_turning_circles(ax, start, end, alpha=0.2)
            
            # Set subplot properties
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Set consistent coordinate range
            margin = max(3, self.turning_radius)
            x_min = min(start[0], end[0]) - margin
            x_max = max(start[0], end[0]) + margin
            y_min = min(start[1], end[1]) - margin
            y_max = max(start[1], end[1]) + margin
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Dubins Path Six Sequences Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _add_direction_arrows(self, ax, path_points: np.ndarray, color: str):
        """Add direction arrows to path"""
        if len(path_points) < 10:
            return
        
        # Add arrows at regular intervals
        arrow_indices = np.linspace(0, len(path_points)-2, 8, dtype=int)
        
        for idx in arrow_indices:
            if idx + 1 < len(path_points):
                start_point = path_points[idx]
                end_point = path_points[idx + 1]
                
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                if abs(dx) > self.eps or abs(dy) > self.eps:
                    ax.annotate('', xy=end_point, xytext=start_point,
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=1.5, alpha=0.7))
    
    def _draw_turning_circles(self, ax, start: Tuple[float, float, float], 
                             end: Tuple[float, float, float], alpha: float = 0.1):
        """Draw turning radius circles for visualization"""
        # Start position circles
        cx1_r = start[0] - self.turning_radius * math.sin(start[2])
        cy1_r = start[1] + self.turning_radius * math.cos(start[2])
        circle1_r = plt.Circle((cx1_r, cy1_r), self.turning_radius, 
                              fill=False, color='blue', alpha=alpha, linestyle='--')
        ax.add_patch(circle1_r)
        
        cx1_l = start[0] + self.turning_radius * math.sin(start[2])
        cy1_l = start[1] - self.turning_radius * math.cos(start[2])
        circle1_l = plt.Circle((cx1_l, cy1_l), self.turning_radius, 
                              fill=False, color='red', alpha=alpha, linestyle='--')
        ax.add_patch(circle1_l)
        
        # End position circles
        cx2_r = end[0] - self.turning_radius * math.sin(end[2])
        cy2_r = end[1] + self.turning_radius * math.cos(end[2])
        circle2_r = plt.Circle((cx2_r, cy2_r), self.turning_radius, 
                              fill=False, color='blue', alpha=alpha, linestyle='--')
        ax.add_patch(circle2_r)
        
        cx2_l = end[0] + self.turning_radius * math.sin(end[2])
        cy2_l = end[1] - self.turning_radius * math.cos(end[2])
        circle2_l = plt.Circle((cx2_l, cy2_l), self.turning_radius, 
                              fill=False, color='red', alpha=alpha, linestyle='--')
        ax.add_patch(circle2_l)
    
    def _draw_vehicle(self, ax, pose: Tuple[float, float, float], 
                     color: str, label: str = '', size: float = 1.0):
        """Draw vehicle with enhanced representation"""
        x, y, theta = pose
        
        # Vehicle dimensions
        length = 1.0 * size
        width = 0.5 * size
        
        # Vehicle vertices (local coordinates)
        vertices = np.array([
            [length, 0],
            [-length/3, width/2],
            [-length/3, -width/2]
        ])
        
        # Rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Transform to global coordinates
        vertices = vertices @ rotation_matrix.T
        vertices[:, 0] += x
        vertices[:, 1] += y
        
        # Draw triangle
        triangle = patches.Polygon(vertices, closed=True, 
                                 facecolor=color, edgecolor='black', 
                                 alpha=0.8, linewidth=2)
        ax.add_patch(triangle)
        
        # Add direction line
        tip_x = x + length * cos_theta
        tip_y = y + length * sin_theta
        ax.plot([x, tip_x], [y, tip_y], color='black', linewidth=2)
        
        # Add label
        if label:
            ax.annotate(label, (x, y), xytext=(8, 8), 
                       textcoords='offset points', fontsize=11, 
                       fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def create_detailed_analysis_table(self, start: Tuple[float, float, float], 
                                     end: Tuple[float, float, float]) -> str:
        """Create detailed path analysis table"""
        paths = self.compute_all_paths(start, end)
        
        # Find shortest path
        shortest_length = float('inf')
        shortest_type = None
        
        for path_type, path_info in paths.items():
            if path_info['feasible'] and path_info['length'] < shortest_length:
                shortest_length = path_info['length']
                shortest_type = path_type
        
        # Create detailed table
        table = "=" * 80 + "\\n"
        table += "Enhanced Dubins Path Six Sequences Analysis\\n"
        table += "=" * 80 + "\\n"
        table += f"Start Pose: ({start[0]:.2f}, {start[1]:.2f}, {math.degrees(start[2]):.1f}°)\\n"
        table += f"End Pose: ({end[0]:.2f}, {end[1]:.2f}, {math.degrees(end[2]):.1f}°)\\n"
        table += f"Turning Radius: {self.turning_radius:.2f}\\n"
        table += f"Euclidean Distance: {math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2):.3f}\\n"
        table += "-" * 80 + "\\n"
        table += f"{'Type':<8} {'Feasible':<10} {'Length':<12} {'Segments (t1,p,t2)':<20} {'Relative':<12}\\n"
        table += "-" * 80 + "\\n"
        
        for path_type in self.path_types:
            path_info = paths[path_type]
            
            feasible_str = "✓ Yes" if path_info['feasible'] else "✗ No"
            
            if path_info['feasible']:
                length_str = f"{path_info['length']:.4f}"
                t1, p, t2 = path_info['segments']
                segments_str = f"({t1:.2f},{p:.2f},{t2:.2f})"
                
                if path_type == shortest_type:
                    relative_str = "★ Optimal"
                else:
                    ratio = path_info['length'] / shortest_length
                    relative_str = f"+{(ratio-1)*100:.1f}%"
            else:
                length_str = "Infeasible"
                segments_str = "(-,-,-)"
                relative_str = "—"
            
            table += f"{path_type:<8} {feasible_str:<10} {length_str:<12} {segments_str:<20} {relative_str:<12}\\n"
        
        table += "-" * 80 + "\\n"
        if shortest_type:
            table += f"Optimal Path: {shortest_type} (Length: {shortest_length:.4f})\\n"
            efficiency = shortest_length / math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            table += f"Path Efficiency: {efficiency:.3f} (vs straight line)\\n"
        else:
            table += "No feasible path found!\\n"
        table += "=" * 80 + "\\n"
        
        return table


def demo_enhanced_dubins_paths():
    """Enhanced demonstration of Dubins path calculation and visualization"""
    print("=== Enhanced Dubins Path Six Sequences Visualization ===\\n")
    
    # Create high-precision visualizer
    visualizer = HighPrecisionDubinsVisualizer(turning_radius=2.5)
    
    # Enhanced test cases
    test_cases = [
        ((0, 0, math.pi/4), (12, 8, -math.pi/4), "Standard Configuration"),
        ((0, 0, 0), (8, 8, math.pi), "Tight Space Configuration"),
        ((0, 0, math.pi/6), (20, 5, -math.pi/3), "Long Distance Configuration"),
        ((0, 0, 0), (0, 10, math.pi), "U-turn Configuration"),
        ((0, 0, math.pi/2), (15, 0, 0), "Complex Maneuver Configuration")
    ]
    
    for i, (start, end, description) in enumerate(test_cases):
        print(f"Test Case {i+1}: {description}")
        
        # Generate detailed analysis table
        table = visualizer.create_detailed_analysis_table(start, end)
        print(table)
        
        # Plot all paths with enhanced visualization
        print("Generating high-precision visualization...")
        visualizer.plot_all_paths(start, end)
        
        if i < len(test_cases) - 1:
            print("Press Enter to continue to next case...")
            input()


if __name__ == "__main__":
    try:
        demo_enhanced_dubins_paths()
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()