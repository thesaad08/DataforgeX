"""
Enhanced Data Visualization Module

A simplified Python module for common data visualization tasks using pandas, matplotlib, and seaborn.
Designed to reduce boilerplate code and make data exploration more accessible with Plotly-like themes.

Author: DataViz Helper
Version: 3.0.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Enhanced themes with better Plotly-like styling
THEMES = {
    "plotly": {
        "style": "whitegrid",
        "colors": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
                  "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
        "bg_color": "white",
        "grid_color": "#E2E2E2",
        "text_color": "#2a3f5f",
        "grid_alpha": 0.3,
        "font_family": "Arial",
        "cmap": "RdYlBu_r"
    },
   "plotly_dark": {
    "style": "darkgrid",
    "colors": [
        "#7AA2F7",  # soft blue
        "#F7768E",  # soft red
        "#9ECE6A",  # green
        "#BB9AF7",  # purple
        "#FF9E64",  # orange
        "#7DCFFF",  # cyan
        "#FCA7EA",  # pink
        "#C0CAF5",  # light violet
        "#9AA5CE",  # muted blue-gray
        "#E0AF68"   # yellow
    ],
    "bg_color": "#1E1E1E",
    "grid_color": "#3A3A3A",
    "text_color": "#E6E6E6",
    "grid_alpha": 0.35,
    "font_family": "Arial",
    "cmap": "viridis"
},

    "seaborn": {
        "style": "darkgrid",
        "colors": sns.color_palette("husl", 10),
        "bg_color": "white",
        "grid_color": "#D0D0D0",
        "text_color": "#333333",
        "grid_alpha": 0.3,
        "font_family": "DejaVu Sans",
        "cmap": "Spectral_r"
    },
    
    "corporate": {
        "style": "ticks",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
        "bg_color": "white",
        "grid_color": "#DDE1E6",
        "text_color": "#1A1A1A",
        "grid_alpha": 0.3,
        "font_family": "Arial",
        "cmap": "Blues"
    }
}

# Global theme setting
current_theme = "plotly"

def set_theme(theme_name="plotly"):
    """
    Set the global theme for all plots with enhanced styling.
    
    Parameters:
    -----------
    theme_name : str, default='plotly'
        Name of the theme. Options: 'plotly', 'plotly_dark', 'seaborn', 
        , 'corporate'
        
    Example:
    --------
    >>> set_theme('plotly_dark')
    """
    global current_theme
    if theme_name in THEMES:
        current_theme = theme_name
        theme_config = THEMES[theme_name]
        
        # Set seaborn style with enhanced parameters
        sns.set_style(theme_config["style"], {
            'grid.color': theme_config["grid_color"],
            'grid.alpha': theme_config["grid_alpha"],
            'axes.facecolor': theme_config["bg_color"],
            'figure.facecolor': theme_config["bg_color"]
        })
        
        # Enhanced matplotlib parameters
        rcParams.update({
            'figure.facecolor': theme_config["bg_color"],
            'axes.facecolor': theme_config["bg_color"],
            'axes.edgecolor': theme_config["grid_color"],
            'axes.labelcolor': theme_config["text_color"],
            'text.color': theme_config["text_color"],
            'xtick.color': theme_config["text_color"],
            'ytick.color': theme_config["text_color"],
            'grid.color': theme_config["grid_color"],
            'grid.alpha': theme_config["grid_alpha"],
            'font.family': theme_config["font_family"],
            'figure.titlesize': 18,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.facecolor': theme_config["bg_color"],
            'legend.edgecolor': theme_config["grid_color"]
        })
        
        # Set color palette
        sns.set_palette(theme_config["colors"])
        print(f"🎨 Theme set to: {theme_name}")
    else:
        available_themes = ", ".join(THEMES.keys())
        print(f"❌ Theme '{theme_name}' not found. Available themes: {available_themes}")

def get_theme_colors(n_colors=5):
    """
    Get color palette for current theme.
    
    Parameters:
    -----------
    n_colors : int, default=5
        Number of colors to return
        
    Returns:
    --------
    list of colors
    """
    theme_config = THEMES[current_theme]
    if n_colors <= len(theme_config["colors"]):
        return theme_config["colors"][:n_colors]
    else:
        # Generate additional colors using seaborn
        return sns.color_palette(theme_config["colors"], n_colors=n_colors)

def apply_plot_style(ax):
    """
    Apply consistent styling to a matplotlib axis.
    """
    theme_config = THEMES[current_theme]
    
    # Remove spines and enhance grid
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(theme_config["grid_color"])
        ax.spines[spine].set_alpha(0.7)
    
    # Enhanced grid
    ax.grid(True, alpha=theme_config["grid_alpha"], linestyle='-', linewidth=0.5)
    
    # Set text colors
    ax.title.set_color(theme_config["text_color"])
    ax.xaxis.label.set_color(theme_config["text_color"])
    ax.yaxis.label.set_color(theme_config["text_color"])
    
    return ax

def plot_histograms(df, figsize=(16, 12), bins=30, alpha=0.8, title=None, 
                   theme=None, show_stats=True):
    """
    Plot enhanced histograms for all numeric columns with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing numeric columns
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    bins : int, default=30
        Number of bins for histograms
    alpha : float, default=0.8
        Transparency level
    title : str, optional
        Overall title for the plot
    theme : str, optional
        Specific theme for this plot
    show_stats : bool, default=True
        Whether to show statistical annotations
        
    Example:
    --------
    >>> plot_histograms(df, figsize=(12, 8), bins=20, theme='plotly')
    """
    if theme:
        set_theme(theme)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ No numeric columns found in the DataFrame.")
        return
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Create subplots with enhanced styling
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    colors = get_theme_colors(len(numeric_cols))
    theme_config = THEMES[current_theme]
    
    # Plot enhanced histograms
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Drop NaN values for the current column
            data = df[col].dropna()
            
            # Create histogram with KDE
            n, bins, patches = axes[i].hist(data, bins=bins, color=colors[i], 
                                          alpha=alpha, density=True,
                                          edgecolor='white', linewidth=1.5)
            
            # Add KDE curve
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[i].plot(x_range, kde(x_range), color='darkred', 
                        linewidth=2, alpha=0.8, label='KDE')
            
            axes[i] = apply_plot_style(axes[i])
            axes[i].set_title(f'Distribution of {col}', fontweight='bold', pad=15, fontsize=13)
            axes[i].set_xlabel(col, fontweight='bold', fontsize=11)
            axes[i].set_ylabel('Density', fontweight='bold', fontsize=11)
            
            # Add statistical annotations
            if show_stats:
                mean_val = data.mean()
                std_val = data.std()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              alpha=0.9, linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', 
                              alpha=0.7, linewidth=1.5)
                axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', 
                              alpha=0.7, linewidth=1.5)
                
                stats_text = f'n: {len(data):,}\nμ: {mean_val:.2f}\nσ: {std_val:.2f}'
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor=theme_config["bg_color"], alpha=0.8),
                           fontsize=9, fontweight='bold')
            
            axes[i].legend(fontsize=9)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    if title:
        plt.suptitle(title, fontsize=20, fontweight='bold', y=0.95, 
                    color=theme_config["text_color"])
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, group_by=None, figsize=(18, 14), title=None, theme=None):
    """
    Plot enhanced boxplots for all numeric columns with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    group_by : str, optional
        Categorical column name to group by
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    title : str, optional
        Overall title for the plot
    theme : str, optional
        Specific theme for this plot
        
    Example:
    --------
    >>> plot_boxplots(df, group_by='category', theme='plotly_dark')
    """
    if theme:
        set_theme(theme)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("❌ No numeric columns found in the DataFrame.")
        return
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    colors = get_theme_colors(10)
    theme_config = THEMES[current_theme]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i] = apply_plot_style(axes[i])
            
            if group_by and group_by in df.columns:
                # Enhanced grouped boxplot
                sns.boxplot(data=df, x=group_by, y=col, ax=axes[i], palette=colors,
                           fliersize=3, linewidth=1.5)
                axes[i].set_title(f'{col} by {group_by}', fontweight='bold', pad=15, fontsize=13)
                axes[i].tick_params(axis='x', rotation=45)
            else:
                # Enhanced single boxplot
                data = df[col].dropna()
                box_plot = axes[i].boxplot(data, patch_artist=True, 
                                         boxprops=dict(facecolor=colors[0], alpha=0.8,
                                         linewidth=1.5),
                                         medianprops=dict(color='red', linewidth=2),
                                         whiskerprops=dict(linewidth=1.5),
                                         capprops=dict(linewidth=1.5),
                                         flierprops=dict(marker='o', markersize=4,
                                         alpha=0.6))
                axes[i].set_title(f'Boxplot of {col}', fontweight='bold', pad=15, fontsize=13)
                axes[i].set_ylabel(col, fontweight='bold', fontsize=11)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    if title:
        plt.suptitle(title, fontsize=20, fontweight='bold', y=0.95,
                    color=theme_config["text_color"])
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, figsize=(12, 10), annot=True, title='Correlation Heatmap', 
                           theme=None, mask_upper=True):
    """
    Plot enhanced correlation heatmap with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    figsize : tuple, default=(14, 12)
        Figure size (width, height)
    annot : bool, default=True
        Whether to annotate correlation values
    title : str, default='Correlation Heatmap'
        Title for the plot
    theme : str, optional
        Specific theme for this plot
    mask_upper : bool, default=True
        Whether to mask upper triangle
        
    Example:
    --------
    >>> plot_correlation_heatmap(df, theme='plotly_dark')
    """
    if theme:
        set_theme(theme)
    
    # Select only numeric columns and compute correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("❌ No numeric columns found for correlation analysis.")
        return
    
    correlation_matrix = numeric_df.corr()
    theme_config = THEMES[current_theme]
    
    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    plt.figure(figsize=figsize)
    
    # Enhanced heatmap with better styling
    sns.heatmap(correlation_matrix, 
                annot=annot, 
                cmap=theme_config["cmap"],
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient',
                         'pad': 0.02},
                linewidths=0.8,
                linecolor=theme_config["bg_color"],
                mask=mask,
                annot_kws={'size': 10, 'weight': 'bold'})
    
    plt.title(title, fontsize=20, fontweight='bold',
             color=theme_config["text_color"])
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_countplots(df, figsize=(16, 12), max_categories=10, title=None, theme=None):
    """
    Plot enhanced count plots for categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    max_categories : int, default=10
        Maximum number of categories to display per column
    title : str, optional
        Overall title for the plot
    theme : str, optional
        Specific theme for this plot
        
    Example:
    --------
    >>> plot_countplots(df, max_categories=6, theme='ggplot')
    """
    if theme:
        set_theme(theme)
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("❌ No categorical columns found in the DataFrame.")
        return
    
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    colors = get_theme_colors(max_categories)
    theme_config = THEMES[current_theme]
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            axes[i] = apply_plot_style(axes[i])
            
            # Get value counts and limit to top categories
            value_counts = df[col].value_counts().head(max_categories)
            
            if len(value_counts) > 0:
                # Create horizontal bar plot
                bars = axes[i].barh(range(len(value_counts)), value_counts.values, 
                                   color=colors[:len(value_counts)], alpha=0.85,
                                   edgecolor='white', linewidth=1.5,
                                   height=0.7)
                
                axes[i].set_yticks(range(len(value_counts)))
                axes[i].set_yticklabels(value_counts.index, fontweight='bold')
                axes[i].set_title(f'Distribution of {col}', fontweight='bold', 
                                pad=15, fontsize=13)
                axes[i].set_xlabel('Count', fontweight='bold', fontsize=11)
                
                # Remove y-axis label for cleaner look
                axes[i].set_ylabel('')
                
                # Enhanced value labels
                for j, (bar, count) in enumerate(zip(bars, value_counts.values)):
                    width = bar.get_width()
                    percentage = (count / len(df)) * 100
                    axes[i].text(width + width * 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{count:,} ({percentage:.1f}%)', 
                                ha='left', va='center', fontweight='bold',
                                fontsize=9, color=theme_config["text_color"])
    
    # Hide empty subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    if title:
        plt.suptitle(title, fontsize=20, fontweight='bold', y=0.95,
                    color=theme_config["text_color"])
    
    plt.tight_layout()
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col=None, figsize=(14, 10), 
                 alpha=0.7, title=None, xlabel=None, ylabel=None, 
                 theme=None, show_trend=True, size=80):
    """
    Plot enhanced scatter plot with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    hue_col : str, optional
        Column name for color grouping
    figsize : tuple, default=(14, 10)
        Figure size (width, height)
    alpha : float, default=0.7
        Transparency level
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    theme : str, optional
        Specific theme for this plot
    show_trend : bool, default=True
        Whether to show trend line
    size : int, default=80
        Marker size
        
    Example:
    --------
    >>> plot_scatter(df, 'age', 'income', hue_col='gender', theme='plotly_dark')
    """
    if theme:
        set_theme(theme)
    
    # Check if columns exist
    if x_col not in df.columns or y_col not in df.columns:
        print(f"❌ Error: Columns '{x_col}' or '{y_col}' not found in DataFrame.")
        return
    
    if hue_col and hue_col not in df.columns:
        print(f"⚠️ Warning: Hue column '{hue_col}' not found. Plotting without grouping.")
        hue_col = None
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax = apply_plot_style(ax)
    
    # Remove rows with missing values in relevant columns
    plot_data = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
    
    colors = get_theme_colors(10)
    theme_config = THEMES[current_theme]
    
    if hue_col:
        # Enhanced grouped scatter plot
        unique_categories = plot_data[hue_col].unique()
        color_palette = colors[:len(unique_categories)]
        
        for i, category in enumerate(unique_categories):
            category_data = plot_data[plot_data[hue_col] == category]
            plt.scatter(category_data[x_col], category_data[y_col], 
                       color=color_palette[i], alpha=alpha, s=size,
                       edgecolors='white', linewidth=1, label=category,
                       marker='o')
        
        # Enhanced legend
        plt.legend(title=hue_col, frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.9, loc='best',
                  title_fontproperties={'weight': 'bold'})
    else:
        # Single scatter plot with gradient coloring
        scatter = plt.scatter(plot_data[x_col], plot_data[y_col], 
                             color=colors[0], alpha=alpha, s=size,
                             edgecolors='white', linewidth=1,
                             cmap=theme_config["cmap"] if not hue_col else None)
    
    # Enhanced labels and title
    plt.xlabel(xlabel or x_col, fontweight='bold', fontsize=12, labelpad=10)
    plt.ylabel(ylabel or y_col, fontweight='bold', fontsize=12, labelpad=10)
    plt.title(title or f'{y_col} vs {x_col}', fontsize=18, fontweight='bold', pad=20)
    
    # Add trend line if requested and no hue
    if show_trend and not hue_col:
        z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.9, linewidth=2.5,
                label=f'Trend (r²: {np.corrcoef(plot_data[x_col], plot_data[y_col])[0,1]**2:.3f})')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def list_themes():
    """
    List all available themes with descriptions.
    
    Example:
    --------
    >>> list_themes()
    """
    theme_descriptions = {
        "plotly": "Clean, modern styling inspired by Plotly (Default)",
        "plotly_dark": "Dark version of the Plotly theme",
        "seaborn": "Enhanced seaborn styling with vibrant colors",
        "corporate": "Clean, business-appropriate styling"
    }
    
    print("🎨 Available Themes:")
    print("-" * 50)
    for theme_name in THEMES.keys():
        desc = theme_descriptions.get(theme_name, "No description available")
        current_indicator = " (Current)" if theme_name == current_theme else ""
        print(f"  • {theme_name:<15} {desc}{current_indicator}")
    print(f"\nCurrent theme: {current_theme}")

def quick_eda(df, theme='plotly', figsize=(16, 12)):
    """
    Perform quick exploratory data analysis with multiple visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    theme : str, default='plotly'
        Theme for the plots
    figsize : tuple, default=(16, 12)
        Figure size
        
    Example:
    --------
    >>> quick_eda(df, theme='plotly_dark')
    """
    set_theme(theme)
    
    print("📊 Quick EDA Report")
    print("=" * 50)
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Features: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n" + "=" * 50)
    
    # Basic info
    print("\n1. Dataset Info:")
    print(df.info())
    
    # Numeric summary
    print("\n2. Numeric Columns Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    # Categorical summary
    print("\n3. Categorical Columns Summary:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    # Generate visualizations
    if len(numeric_cols) > 0:
        print("\n4. Generating Visualizations...")
        plot_histograms(df, title="Distribution of Numeric Features", figsize=figsize)
        
        if len(numeric_cols) > 1:
            plot_correlation_heatmap(df, title="Feature Correlation Matrix", figsize=(12, 10))
        
        if len(categorical_cols) > 0:
            # Use first categorical column for grouping if available
            group_col = categorical_cols[0] if len(categorical_cols) > 0 else None
            if group_col and df[group_col].nunique() <= 10:  # Limit to reasonable categories
                plot_boxplots(df, group_by=group_col, 
                            title=f"Boxplots by {group_col}", figsize=figsize)
    
    if len(categorical_cols) > 0:
        plot_countplots(df, title="Distribution of Categorical Features", figsize=figsize)

# Initialize with default theme
set_theme('plotly')

# Example usage and test function
def _example_usage():
    """
    Enhanced example usage of the data_viz module with themes.
    """
    # Create more realistic sample data
    np.random.seed(42)
    n_samples = 300
    
    sample_data = pd.DataFrame({
        'age': np.random.normal(35, 8, n_samples),
        'income': np.random.lognormal(10.5, 0.4, n_samples),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'experience': np.random.gamma(2, 2, n_samples),
        'category': np.random.choice(['Premium', 'Standard', 'Basic', 'Trial'], n_samples, p=[0.1, 0.5, 0.3, 0.1]),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'satisfaction': np.random.choice(['Very High', 'High', 'Medium', 'Low'], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    })
    
    # Add some missing values
    sample_data.loc[::20, 'income'] = np.nan
    sample_data.loc[::25, 'score'] = np.nan
    
    print("📈 Enhanced Data Visualization Demo")
    print("=" * 60)
    print("Sample Data Overview:")
    print(sample_data.head())
    
    # List available themes
    print("\n" + "=" * 60)
    list_themes()
    
    # Demo different themes
    themes_to_demo = ['plotly', 'plotly_dark','corporate']
    
    for theme in themes_to_demo:
        print(f"\n{'='*60}")
        print(f"🎨 DEMONSTRATING {theme.upper()} THEME")
        print(f"{'='*60}")
        
        # Histograms
        plot_histograms(sample_data, title=f"Distributions - {theme.title()} Theme", 
                       theme=theme, figsize=(14, 10))
        
        # Correlation heatmap
        # if len(sample_data.select_dtypes(include=[np.number]).columns) > 1:
        #     plot_correlation_heatmap(sample_data, title=f"Correlations - {theme.title()} Theme",
        #                            theme=theme, figsize=(10, 8))
        
        # # Count plots
        # plot_countplots(sample_data, title=f"Category Distributions - {theme.title()} Theme",
        #                theme=theme, figsize=(14, 8))
        
        # # Scatter plot
        # plot_scatter(sample_data, 'age', 'income', hue_col='department',
        #             title=f'Income vs Age - {theme.title()} Theme', theme=theme)

if __name__ == "__main__":
    _example_usage()