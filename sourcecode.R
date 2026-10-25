Source scripts for the TaUTP18 project, including R and Python scripts for figure plots and modeling as described below:  

#fig.2A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure fonts and visualization settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("ticks")

# Specify global font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 20
})


def read_marker_bands(marker_file):
    """
    Read spectral band positions to mark from a file
    """
    try:
        if marker_file.endswith('.txt'):
            # Read wavelengths from the text file (one value per line)
            with open(marker_file, 'r') as f:
                marker_bands = [float(line.strip()) for line in f if line.strip()]
        elif marker_file.endswith('.csv'):
            # Read the first column of the CSV file
            df = pd.read_csv(marker_file)
            marker_bands = df.iloc[:, 0].values.tolist()
        elif marker_file.endswith(('.xlsx', '.xls')):
            # Read the first column of the Excel file
            df = pd.read_excel(marker_file)
            marker_bands = df.iloc[:, 0].values.tolist()
        else:
            print(f"Unsupported file format: {marker_file}")
            return []

        # Convert values to numeric format and sort them
        marker_bands = sorted([float(x) for x in marker_bands if pd.notna(x)])
        print(f"Successfully loaded {len(marker_bands)} marker bands: {marker_bands}")
        return marker_bands

    except Exception as e:
        print(f"Error reading the marker-band file: {str(e)}")
        return []


def plot_spectral_reflectance_with_shading(reflectance_file, wavelength_file,
                                           marker_file=None,
                                           output_file='Spectral_Reflectance_Plot.png'):
    """
    Plot spectral reflectance curves with standard-deviation bands and vertical dashed lines at specified wavelengths

    Parameters:
        reflectance_file: Path to the reflectance data file
        wavelength_file: Path to the wavelength data file
        marker_file: Path to the marker-band file
        output_file: Output image filename
    """

    # Import the data
    if reflectance_file.endswith('.csv'):
        reflectance_df = pd.read_csv(reflectance_file)
    else:
        reflectance_df = pd.read_excel(reflectance_file)

    if wavelength_file.endswith('.csv'):
        wavelength_df = pd.read_csv(wavelength_file)
    else:
        wavelength_df = pd.read_excel(wavelength_file)

    # Extract wavelength data
    wavelengths = wavelength_df.iloc[:, 1].values[:314]

    # Extract reflectance data
    reflectance_columns = reflectance_df.columns[1:315]
    reflectance_data = reflectance_df[reflectance_columns].values

    # Extract treatment information
    if len(reflectance_df.columns) > 315:
        treatment_col = reflectance_df.columns[315]
    else:
        treatment_col = reflectance_df.columns[-1]

    treatments = reflectance_df[treatment_col].values

    # Group by treatment
    unique_treatments = np.unique(treatments)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'C': '#1f77b4', 'H': '#d62728'}

    for treatment in unique_treatments:
        mask = treatments == treatment
        treatment_data = reflectance_data[mask, :]

        # Compute the mean and standard deviation
        mean_reflectance = np.mean(treatment_data, axis=0)
        std_reflectance = np.std(treatment_data, axis=0)

        # Plot the mean curve
        ax.plot(wavelengths, mean_reflectance,
                color=colors.get(treatment, 'gray'),
                linewidth=3.5,
                label=f' {treatment} (mean)')

        # Plot the mean reflectance with a standard-deviation envelope
        ax.fill_between(wavelengths,
                        mean_reflectance - std_reflectance,
                        mean_reflectance + std_reflectance,
                        color=colors.get(treatment, 'gray'),
                        alpha=0.2,
                        label=f' {treatment} (±std)')

    # Read reference wavelengths and add vertical dashed markers
    if marker_file:
        marker_bands = read_marker_bands(marker_file)
        if marker_bands:
            for band in marker_bands:
                # Verify that the band falls within the wavelength range
                if wavelengths.min() <= band <= wavelengths.max():
                    ax.axvline(x=band, color='gray', linestyle='--',
                               linewidth=1.5, alpha=0.7, zorder=1)
                else:
                    print(f"Warning: band {band} nm is outside the wavelength range [{wavelengths.min():.1f}, {wavelengths.max():.1f}] within")

    # Configure plot properties
    ax.set_xlabel('Wavelength (nm)', fontsize=22, fontweight='bold')
    ax.set_ylabel('VIS_A', fontsize=22, fontweight='bold')

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Set x-axis ticks and display only 10 labels
    num_ticks = 10
    tick_indices = np.linspace(0, len(wavelengths) - 1, num_ticks, dtype=int)
    ax.set_xticks(wavelengths[tick_indices])
    ax.set_xticklabels([f'{w:.1f}' for w in wavelengths[tick_indices]], rotation=45)
    # Make y-axis tick labels bold
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Make legend text bold
    legend = ax.legend(loc='best', fontsize=18, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot with error bands saved as: {output_file}")


def validate_band_names(reflectance_file, wavelength_file):
    """
    Verify that reflectance column names match wave band names
    """
    print("Checking band-name consistency...")

    # Import the data
    if reflectance_file.endswith('.csv'):
        reflectance_df = pd.read_csv(reflectance_file)
    else:
        reflectance_df = pd.read_excel(reflectance_file)

    if wavelength_file.endswith('.csv'):
        wavelength_df = pd.read_csv(wavelength_file)
    else:
        wavelength_df = pd.read_excel(wavelength_file)

    # Get reflectance band column names (2-487 columns)
    reflectance_band_names = list(reflectance_df.columns[1:314])

    # Get wavelength band names (first column)
    wavelength_band_names = list(wavelength_df.iloc[:, 0].astype(str))

    print(f"Number of reflectance bands: {len(reflectance_band_names)}")
    print(f"Number of wavelength bands: {len(wavelength_band_names)}")

    # Check for matching band names
    common_bands = set(reflectance_band_names) & set(wavelength_band_names)
    print(f"Number of shared bands: {len(common_bands)}")

    if len(common_bands) < len(reflectance_band_names):
        print("Warning:  Not all reflectance band names were found in the wavelength data")
        missing_in_wavelength = set(reflectance_band_names) - set(wavelength_band_names)
        if missing_in_wavelength:
            print(f"Bands missing from wavelength data: {list(missing_in_wavelength)[:10]}...")

    return True


# Main program
if __name__ == "__main__":
    # Set file paths 
    reflectance_file = r"CH_A.xlsx"
    wavelength_file = r"wave_length.xlsx"

    marker_file = r"Selected_band.xlsx"  

    try:
        # First verify that band names match
        validate_band_names(reflectance_file, wavelength_file)

        # Plot variability envelopes and reference-band markers
        plot_spectral_reflectance_with_shading(
            reflectance_file,
            wavelength_file,
            marker_file=marker_file,  
            output_file='A_VIS1.png'
        )
    except Exception as e:
        print(f"Error executing the program: {str(e)}")
        import traceback

        traceback.print_exc()


#fig.2B
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure fonts and visualization settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("ticks")

# Specify global font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 20
})


def read_marker_bands(marker_file):
    """
    Read spectral band positions to mark from a file
    Supports TXT, CSV, and XLSX formats
    File format：One wavelength value per line
    """
    try:
        if marker_file.endswith('.txt'):
            # Read wavelengths from the text file (one value per line)
            with open(marker_file, 'r') as f:
                marker_bands = [float(line.strip()) for line in f if line.strip()]
        elif marker_file.endswith('.csv'):
            # Read the first column of the CSV file
            df = pd.read_csv(marker_file)
            marker_bands = df.iloc[:, 0].values.tolist()
        elif marker_file.endswith(('.xlsx', '.xls')):
            # Read the first column of the Excel file
            df = pd.read_excel(marker_file)
            marker_bands = df.iloc[:, 0].values.tolist()
        else:
            print(f"Unsupported file format: {marker_file}")
            return []

        # Convert values to numeric format and sort them
        marker_bands = sorted([float(x) for x in marker_bands if pd.notna(x)])
        print(f"Successfully loaded {len(marker_bands)} marker bands: {marker_bands}")
        return marker_bands

    except Exception as e:
        print(f"Error reading the marker-band file: {str(e)}")
        return []


def plot_spectral_reflectance_with_shading(reflectance_file, wavelength_file,
                                           marker_file=None,
                                           output_file='Spectral_Reflectance_Plot.png'):
    """
    Plot spectral reflectance curves with standard-deviation bands and vertical dashed lines at specified wavelengths
    """

    # Import the data
    if reflectance_file.endswith('.csv'):
        reflectance_df = pd.read_csv(reflectance_file)
    else:
        reflectance_df = pd.read_excel(reflectance_file)

    if wavelength_file.endswith('.csv'):
        wavelength_df = pd.read_csv(wavelength_file)
    else:
        wavelength_df = pd.read_excel(wavelength_file)

    # Extract wavelength data
    wavelengths = wavelength_df.iloc[:, 1].values[:172]

    # Extract reflectance data
    reflectance_columns = reflectance_df.columns[1:173]
    reflectance_data = reflectance_df[reflectance_columns].values

    # Extract treatment information
    if len(reflectance_df.columns) > 173:
        treatment_col = reflectance_df.columns[173]
    else:
        treatment_col = reflectance_df.columns[-1]

    treatments = reflectance_df[treatment_col].values

    # Group by treatment
    unique_treatments = np.unique(treatments)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'C': '#1f77b4', 'H': '#d62728'}

    for treatment in unique_treatments:
        mask = treatments == treatment
        treatment_data = reflectance_data[mask, :]

        # Compute the mean and standard deviation
        mean_reflectance = np.mean(treatment_data, axis=0)
        std_reflectance = np.std(treatment_data, axis=0)

        # Plot the mean curve
        ax.plot(wavelengths, mean_reflectance,
                color=colors.get(treatment, 'gray'),
                linewidth=3.5,
                label=f' {treatment} (mean)')

        # Plot the mean reflectance with a standard-deviation envelope
        ax.fill_between(wavelengths,
                        mean_reflectance - std_reflectance,
                        mean_reflectance + std_reflectance,
                        color=colors.get(treatment, 'gray'),
                        alpha=0.2,
                        label=f' {treatment} (±std)')

    # Read reference wavelengths and add vertical dashed markers
    if marker_file:
        marker_bands = read_marker_bands(marker_file)
        if marker_bands:
            for band in marker_bands:
                # Verify that the band falls within the wavelength range
                if wavelengths.min() <= band <= wavelengths.max():
                    ax.axvline(x=band, color='gray', linestyle='--',
                               linewidth=1.5, alpha=0.7, zorder=1)
                else:
                    print(f"Warning: band {band} nm is outside the wavelength range [{wavelengths.min():.1f}, {wavelengths.max():.1f}] within")

    # Configure plot properties
    ax.set_xlabel('Wavelength (nm)', fontsize=22, fontweight='bold')
    ax.set_ylabel('NIR_A', fontsize=22, fontweight='bold')

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Set x-axis ticks and display only 10 labels
    num_ticks = 10
    tick_indices = np.linspace(0, len(wavelengths) - 1, num_ticks, dtype=int)
    ax.set_xticks(wavelengths[tick_indices])
    ax.set_xticklabels([f'{w:.1f}' for w in wavelengths[tick_indices]], rotation=45)
    # Make y-axis tick labels bold
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Make legend text bold
    legend = ax.legend(loc='best', fontsize=18, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot with error bands saved as: {output_file}")


def validate_band_names(reflectance_file, wavelength_file):
    """
    Verify that reflectance column names match wavelength band names
    """
    print("Checking band-name consistency...")

    # Import the data
    if reflectance_file.endswith('.csv'):
        reflectance_df = pd.read_csv(reflectance_file)
    else:
        reflectance_df = pd.read_excel(reflectance_file)

    if wavelength_file.endswith('.csv'):
        wavelength_df = pd.read_csv(wavelength_file)
    else:
        wavelength_df = pd.read_excel(wavelength_file)

    # Get reflectance band column names
    reflectance_band_names = list(reflectance_df.columns[1:173])

    # Get wavelength band names
    wavelength_band_names = list(wavelength_df.iloc[:, 0].astype(str))

    print(f"Number of reflectance bands: {len(reflectance_band_names)}")
    print(f"Number of wavelength bands: {len(wavelength_band_names)}")

    # Check for matching band names
    common_bands = set(reflectance_band_names) & set(wavelength_band_names)
    print(f"Number of shared bands: {len(common_bands)}")

    if len(common_bands) < len(reflectance_band_names):
        print("Warning: Not all reflectance band names were found in the wavelength data")
        missing_in_wavelength = set(reflectance_band_names) - set(wavelength_band_names)
        if missing_in_wavelength:
            print(f"Bands missing from wavelength data: {list(missing_in_wavelength)[:10]}...")

    return True


# Main program
if __name__ == "__main__":
    # Set file paths 
    reflectance_file = r"CH_A.xlsx"
    wavelength_file = r"wave_length.xlsx"

    marker_file = r"Selected_band.xlsx"  

    try:
        # First verify that band names match
        validate_band_names(reflectance_file, wavelength_file)

        # Plot variability envelopes and reference-band markers
        plot_spectral_reflectance_with_shading(
            reflectance_file,
            wavelength_file,
            marker_file=marker_file,  
            output_file='A_NIR1.png'
        )
    except Exception as e:
        print(f"Error executing the program: {str(e)}")
        import traceback

        traceback.print_exc()


#fig.2C
library(tidyverse)
library(openxlsx)
library(ggplot2)
library(patchwork)
library(ggtext)  
pca_data <- read.xlsx("CH_A.xlsx", 1)
pca_matrix <- pca_data[, -(1:2)]
pca_result <- prcomp(pca_matrix, scale. = TRUE)

pca_df <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Treatment = pca_data$Treatment
)

var_explained <- summary(pca_result)$importance[2, ] * 100
colors <- c("C" = "#1f77b4", "H" = "#d62728")
p_main <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Treatment)) +
  geom_point(size = 3.5, alpha = 0.8) +
  stat_ellipse(level = 0.95, linewidth = 1.5) +
  scale_color_manual(values = colors) +
  
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.15))) +
  scale_x_continuous(expand = expansion(mult = c(0.1, 0.1))) +
  labs(
    x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 16) +
  theme(
    text = element_text(face = "bold"),
    plot.title = element_markdown(size = 24, hjust = 0.5, margin = margin(b = 5)),
    plot.subtitle = element_markdown(size = 22, hjust = 0.5, margin = margin(b = 15)),
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20, face = "bold", color = "black"),
    panel.border = element_rect(color = "black", linewidth = 1),
    legend.position = "none"
  )


p_density <- ggplot(pca_df, aes(x = PC2, fill = Treatment, color = Treatment)) +
  geom_density(alpha = 0.5, linewidth = 1.2) +
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  scale_x_continuous(expand = expansion(mult = c(0.15, 0.15))) +
  coord_flip() +
  theme_void() +
  theme(legend.position = "none")
combined <- p_main + p_density + 
  plot_layout(ncol = 2, widths = c(4, 1))


ggsave("pca_vis_plus1.png",
       plot = combined, 
       width = 9, 
       height = 7, 
       dpi = 300)

#fig.2D
library(tidyverse)
library(openxlsx)
library(ggplot2)
library(patchwork)
library(ggtext)  

pca_data <- read.xlsx("CH_A.xlsx", 2)


pca_matrix <- pca_data[, -(1:2)]
pca_result <- prcomp(pca_matrix, scale. = TRUE)


pca_df <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Treatment = pca_data$Treatment
)


var_explained <- summary(pca_result)$importance[2, ] * 100


colors <- c("C" = "#1f77b4", "H" = "#d62728")


p_main <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Treatment)) +
  geom_point(size = 3.5, alpha = 0.8) +
  stat_ellipse(level = 0.95, linewidth = 1.5) +
  scale_color_manual(values = colors) +

  scale_y_continuous(expand = expansion(mult = c(0.15, 0.15))) +
  scale_x_continuous(expand = expansion(mult = c(0.1, 0.1))) +
  labs(
    x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 16) +
  theme(
    text = element_text(face = "bold"),
    plot.title = element_markdown(size = 24, hjust = 0.5, margin = margin(b = 5)),
    plot.subtitle = element_markdown(size = 22, hjust = 0.5, margin = margin(b = 15)),
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20, face = "bold", color = "black"),
    panel.border = element_rect(color = "black", linewidth = 1),
    legend.position = "none"
  )


p_density <- ggplot(pca_df, aes(x = PC2, fill = Treatment, color = Treatment)) +
  geom_density(alpha = 0.5, linewidth = 1.2) +
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  scale_x_continuous(expand = expansion(mult = c(0.15, 0.15))) +
  coord_flip() +
  theme_void() +
  theme(legend.position = "none")

combined <- p_main + p_density + 
  plot_layout(ncol = 2, widths = c(4, 1))

ggsave("pca_nir_plus1.png",
       plot = combined, 
       width = 9, 
       height = 7, 
       dpi = 300)


#fig.2E
library(ComplexHeatmap)
library(circlize)
library(openxlsx)
library(ggplot2)
library(dplyr)

data <- read.xlsx("CH_A.xlsx", 3)
data <- data[-c(1:2)]

data_C <- data[data$Treatment == "C", 2:465]
data_H <- data[data$Treatment == "H", 2:465]
band_names <- colnames(data_C)

mat_C <- as.matrix(data_C)
mat_H <- as.matrix(data_H)
mat_combined <- rbind(mat_C, mat_H)
mat_scaled <- scale(mat_combined, center = TRUE, scale = TRUE)

n_C <- nrow(mat_C)
mat_C_scaled <- mat_scaled[1:n_C, ]
mat_H_scaled <- mat_scaled[(n_C+1):nrow(mat_scaled), ]

set.seed(123)
n_clusters <- 4
km_result <- kmeans(t(mat_scaled), centers = n_clusters, nstart = 25, iter.max = 100)
column_km <- km_result$cluster

cluster_wavelength_info <- data.frame()

for(i in 1:n_clusters) {
  cluster_indices <- which(column_km == i)
  cluster_wavelengths <- as.numeric(band_names[cluster_indices])
  
  cluster_info <- data.frame(
    Cluster_ID = i,
    N_Bands = length(cluster_wavelengths),
    Min_Wavelength = min(cluster_wavelengths),
    Max_Wavelength = max(cluster_wavelengths),
    Mean_Wavelength = mean(cluster_wavelengths)
  )
  
  cluster_wavelength_info <- rbind(cluster_wavelength_info, cluster_info)
}

cluster_wavelength_info <- cluster_wavelength_info[order(cluster_wavelength_info$Mean_Wavelength), ]

desired_order <- cluster_wavelength_info$Cluster_ID
order_bands <- order(match(column_km, desired_order))
column_km_ordered <- column_km[order_bands]

col_fun <- colorRamp2(c(min(mat_scaled), 0, max(mat_scaled)), 
                      c("#0571B0", "white", "#CA0020"))

cluster_labels <- sapply(1:n_clusters, function(i) {
  cluster_id <- desired_order[i]
  info <- cluster_wavelength_info[cluster_wavelength_info$Cluster_ID == cluster_id, ]
  sprintf("Cluster %d\n(%d-%d nm)", i, 
          round(info$Min_Wavelength), 
          round(info$Max_Wavelength))
})

ht_C <- Heatmap(
  mat_C_scaled[, order_bands],
  name = "Control",
  col = col_fun,
  cluster_rows = TRUE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = FALSE,
  column_split = factor(column_km_ordered, levels = unique(column_km_ordered)),
  column_title = cluster_labels,
  column_title_gp = gpar(fontsize = 10, fontface = "bold"),
  column_gap = unit(2, "mm"),
  border = FALSE,
  row_title = "Control",
  row_title_gp = gpar(fontsize = 14, fontface = "bold"),
  heatmap_legend_param = list(
    title = "Standardized\nReflectance",
    title_gp = gpar(fontsize = 12, fontface = "bold"),
    direction = "vertical",
    border = "black"
  )
)

ht_H <- Heatmap(
  mat_H_scaled[, order_bands],
  name = "Drought",
  col = col_fun,
  cluster_rows = TRUE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = FALSE,
  column_split = factor(column_km_ordered, levels = unique(column_km_ordered)),
  column_title = cluster_labels,
  column_title_gp = gpar(fontsize = 10, fontface = "bold"),
  column_gap = unit(2, "mm"),
  border = FALSE,
  row_title = "Drought",
  row_title_gp = gpar(fontsize = 14, fontface = "bold"),
  show_heatmap_legend = FALSE
)

ht_list <- ht_C %v% ht_H

pdf("heatmap_4clusters.pdf", width = 14, height = 12)
draw(ht_list, 
     column_title = "Spectral Bands (4 Clusters)",
     column_title_gp = gpar(fontsize = 14, fontface = "bold"),
     column_title_side = "top",
     heatmap_legend_side = "right")
dev.off()

png("heatmap_4clusters.png", width = 2000, height = 2400, res = 300)
draw(ht_list, 
     column_title = "Spectral Bands (4 Clusters)",
     column_title_gp = gpar(fontsize = 14, fontface = "bold"),
     column_title_side = "top",
     heatmap_legend_side = "right")
dev.off()


#fig.2F
cluster_mapping <- setNames(1:n_clusters, desired_order)
column_km_renumbered <- cluster_mapping[as.character(column_km)]

prepare_plot_data <- function(mat, cluster_info, treatment_name) {
  plot_data_list <- list()
  for(cluster_id in 1:n_clusters) {
    cluster_indices <- which(cluster_info == cluster_id)
    if(length(cluster_indices) > 0) {
      cluster_mat <- mat[, cluster_indices, drop = FALSE]
      cluster_wavelengths <- as.numeric(band_names[cluster_indices])
      means <- colMeans(cluster_mat)
      sds <- apply(cluster_mat, 2, sd)
      ses <- sds / sqrt(nrow(cluster_mat))
      df <- data.frame(
        Wavelength = cluster_wavelengths,
        Mean = means,
        SE = ses,
        Lower = means - ses,
        Upper = means + ses,
        Treatment = treatment_name,
        Cluster = cluster_id
      )
      plot_data_list[[cluster_id]] <- df
    }
  }
  return(do.call(rbind, plot_data_list))
}

data_C_plot <- prepare_plot_data(mat_C_scaled, column_km_renumbered, "Control")
data_H_plot <- prepare_plot_data(mat_H_scaled, column_km_renumbered, "Drought")
combined_plot_data <- rbind(data_C_plot, data_H_plot)

combined_plot_data$Cluster_Label <- factor(
  combined_plot_data$Cluster,
  levels = 1:n_clusters,
  labels = sapply(1:n_clusters, function(i) {
    info <- cluster_wavelength_info[cluster_wavelength_info$Cluster_ID == desired_order[i], ]
    sprintf("Cluster %d\n%d-%d nm", i, 
            round(info$Min_Wavelength), 
            round(info$Max_Wavelength))
  })
)

theme_journal <- theme_bw(base_size = 14) +
  theme(
   
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold", 
      size = 16,
      color = "black",
      margin = margin(b = 10)
    ),
    
    axis.title.x = element_text(
      face = "bold", 
      size = 14,
      color = "black",
      margin = margin(t = 10)
    ),
    axis.title.y = element_text(
      face = "bold", 
      size = 14,
      color = "black",
      margin = margin(r = 10)
    ),
    
    axis.text.x = element_text(
      face = "bold",
      size = 12,
      color = "black",
      angle = 0
    ),
    axis.text.y = element_text(
      face = "bold",
      size = 12,
      color = "black"
    ),

    strip.text = element_text(
      face = "bold",
      size = 13,
      color = "black"
    ),
    strip.background = element_rect(
      fill = "grey90",
      color = "black",
      size = 1
    ),

    legend.title = element_text(
      face = "bold",
      size = 13,
      color = "black"
    ),
    legend.text = element_text(
      face = "bold",
      size = 12,
      color = "black"
    ),
    legend.position = "bottom",
    legend.background = element_rect(
      fill = "white",
      #color = "black",
      size = 0
    ),
    legend.key.size = unit(1.2, "cm"),
 
    panel.grid.major = element_line(
      color = "grey85",
      size = 0.5
    ),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(
      color = "black",
      fill = NA,
      size = 1.2
    ),
    panel.background = element_rect(
      fill = "white"
    ),

    plot.background = element_rect(
      fill = "white",
      color = NA
    ),

    panel.spacing = unit(1, "lines")
  )

p_all <- ggplot(combined_plot_data, aes(x = Wavelength, y = Mean, 
                                        color = Treatment, fill = Treatment)) +
  geom_line(size = 1.2, alpha = 0.9) +
 
  geom_ribbon(aes(ymin = Lower, ymax = Upper), 
              alpha = 0.2, 
              color = NA) +

  geom_point(size = 0.8, alpha = 0.6) +

  facet_wrap(~Cluster_Label, ncol = 2, scales = "free_x") +

  scale_color_manual(
    values = c("Control" = "#0571B0", "Drought" = "#CA0020"),
    name = "Treatment",
    labels = c("Control", "Drought")
  ) +
  scale_fill_manual(
    values = c("Control" = "#0571B0", "Drought" = "#CA0020"),
    name = "Treatment",
    labels = c("Control", "Drought")
  ) +

  labs(
    title = expression(bold("Spectral Reflectance Patterns by Cluster")),
    x = expression(bold("Wavelength (nm)")),
    y = expression(bold("Standardized Reflectance"))
  ) +

  theme_journal +

  guides(
    color = guide_legend(
      override.aes = list(size = 3, alpha = 1),
      nrow = 1
    ),
    fill = guide_legend(
      override.aes = list(alpha = 0.3),
      nrow = 1
    )
  )


#fig.2G
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
import warnings

warnings.filterwarnings('ignore')

# Configure fonts and visualization settings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Specify global font sizes
plt.rcParams.update({
    'font.size': 10,  # Global font size
    'axes.titlesize': 12,  # Title font size
    'axes.labelsize': 11,  # Axis label font size
    'xtick.labelsize': 9,  # X-axis tick label font size
    'ytick.labelsize': 9,  # y-axis tick label font size
    'legend.fontsize': 10,  # Legend font size
    'figure.titlesize': 14  # Figure title font size
})


def calculate_coefficient_of_variation(reflectance_file, treatment_col='treatment'):
    """
    Calculate the coefficient of variation for each band in groups C and H(CV)
    CV = standard deviation / mean * 100%
    """

    # Read reflectance data
    if reflectance_file.endswith('.csv'):
        reflectance_df = pd.read_csv(reflectance_file)
    else:
        reflectance_df = pd.read_excel(reflectance_file)

    print(f"Data shape: {reflectance_df.shape}")
    print(f"Treatment column: {treatment_col}")

    # Check whether the treatment column exists
    if treatment_col not in reflectance_df.columns:
        print(f"Warning: column '{treatment_col}' does not exist in the data")
        # Attempt to identify the treatment column automatically
        potential_cols = ['treatment', 'Treatment', 'group', 'Group', 'class', 'Class']
        for col in potential_cols:
            if col in reflectance_df.columns:
                treatment_col = col
                print(f"Automatically selected treatment column: {treatment_col}")
                break

    # Extract treatment information
    treatments = reflectance_df[treatment_col].values
    unique_treatments = np.unique(treatments)
    print(f"Treatment types: {unique_treatments}")

    # Identify numeric reflectance columns, excluding treatment metadata
    reflectance_cols = []
    for col in reflectance_df.columns:
        if col != treatment_col and pd.api.types.is_numeric_dtype(reflectance_df[col]):
            reflectance_cols.append(col)

    print(f"Found {len(reflectance_cols)} bands")

    # Prepare the results Data Frame
    cv_results = pd.DataFrame({
        'band': reflectance_cols,
        'wavelength': pd.to_numeric(reflectance_cols, errors='coerce'),
        'cv_C': np.nan,
        'cv_H': np.nan,
        'mean_C': np.nan,
        'mean_H': np.nan,
        'std_C': np.nan,
        'std_H': np.nan
    })

    # Calculate the coefficient of variation for each band
    for i, band in enumerate(reflectance_cols):
        # Separate data for groups C and H
        c_data = reflectance_df.loc[treatments == 'C', band].dropna().values
        h_data = reflectance_df.loc[treatments == 'H', band].dropna().values

        # Compute summary statistics for group C
        if len(c_data) > 0:
            mean_c = np.mean(c_data)
            std_c = np.std(c_data, ddof=1)
            cv_c = (std_c / mean_c * 100) if mean_c != 0 else np.nan
        else:
            mean_c, std_c, cv_c = np.nan, np.nan, np.nan

        # Compute summary statistics for group H
        if len(h_data) > 0:
            mean_h = np.mean(h_data)
            std_h = np.std(h_data, ddof=1)
            cv_h = (std_h / mean_h * 100) if mean_h != 0 else np.nan
        else:
            mean_h, std_h, cv_h = np.nan, np.nan, np.nan

        # Store results
        cv_results.loc[i, 'mean_C'] = mean_c
        cv_results.loc[i, 'mean_H'] = mean_h
        cv_results.loc[i, 'std_C'] = std_c
        cv_results.loc[i, 'std_H'] = std_h
        cv_results.loc[i, 'cv_C'] = cv_c
        cv_results.loc[i, 'cv_H'] = cv_h

    # Sort by wavelength
    if cv_results['wavelength'].notna().any():
        cv_results = cv_results.sort_values('wavelength')

    # Compute summary statistics
    cv_stats = {
        'cv_C_mean': cv_results['cv_C'].mean(),
        'cv_H_mean': cv_results['cv_H'].mean(),
        'cv_C_median': cv_results['cv_C'].median(),
        'cv_H_median': cv_results['cv_H'].median(),
        'cv_C_min': cv_results['cv_C'].min(),
        'cv_H_min': cv_results['cv_H'].min(),
        'cv_C_max': cv_results['cv_C'].max(),
        'cv_H_max': cv_results['cv_H'].max(),
        'cv_C_std': cv_results['cv_C'].std(),
        'cv_H_std': cv_results['cv_H'].std(),
    }

    print(f"Calculation completed！")
    print(f"C group mean CV: {cv_stats['cv_C_mean']:.2f}%")
    print(f"H group mean CV: {cv_stats['cv_H_mean']:.2f}%")
    print(f"C group CV range: {cv_stats['cv_C_min']:.2f}% - {cv_stats['cv_C_max']:.2f}%")
    print(f"H group CV range: {cv_stats['cv_H_min']:.2f}% - {cv_stats['cv_H_max']:.2f}%")

    return cv_results, cv_stats


def plot_cv_line_plot(cv_results, output_file='CV_Line_Plot.png'):
    """
    Plot coefficient-of-variation curves for groups C and H
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get wavelength data
    wavelengths = cv_results['wavelength'].values
    cv_c = cv_results['cv_C'].values
    cv_h = cv_results['cv_H'].values

    # Plot line curves
    ax.plot(wavelengths, cv_c, color='#1f77b4', linewidth=1.5, label='Control (C)', alpha=0.8)
    ax.plot(wavelengths, cv_h, color='#d62728', linewidth=1.5, label='Treatment (H)', alpha=0.8)

    # Add shaded regions to represent variability
    ax.fill_between(wavelengths, cv_c, cv_h, where=(cv_c >= cv_h),
                    color='#1f77b4', alpha=0.2, label='C ≥ H')
    ax.fill_between(wavelengths, cv_c, cv_h, where=(cv_c < cv_h),
                    color='#d62728', alpha=0.2, label='C < H')

    # Compute moving averages to highlight spectral trends
    window_size = 10
    if len(wavelengths) > window_size:
        cv_c_smooth = pd.Series(cv_c).rolling(window=window_size, center=True).mean()
        cv_h_smooth = pd.Series(cv_h).rolling(window=window_size, center=True).mean()
        ax.plot(wavelengths, cv_c_smooth, color='#1f77b4', linewidth=2.5, label='C (smoothed)')
        ax.plot(wavelengths, cv_h_smooth, color='#d62728', linewidth=2.5, label='H (smoothed)')

    # Set Axis labels
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')

    # Set the title
    ax.set_title('Coefficient of Variation by Wavelength: C vs H Groups',
                 fontsize=14, fontweight='bold', pad=15)

    # Configure the grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Configure borders
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Add a legend
    ax.legend(loc='upper right', fontsize=10, frameon=True,
              fancybox=True, framealpha=0.9, edgecolor='black')

    # Add a statistical summary box
    stats_text = (f'C mean CV: {cv_results["cv_C"].mean():.1f}%\n'
                  f'H mean CV: {cv_results["cv_H"].mean():.1f}%\n'
                  f'C median CV: {cv_results["cv_C"].median():.1f}%\n'
                  f'H median CV: {cv_results["cv_H"].median():.1f}%')

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.9, edgecolor='black'))

    # Set x-axis ticks (show selected ticks only)
    if len(wavelengths) > 10:
        tick_indices = np.linspace(0, len(wavelengths) - 1, 8, dtype=int)
        ax.set_xticks(wavelengths[tick_indices])
        ax.set_xticklabels([f'{w:.0f}' for w in wavelengths[tick_indices]], rotation=45)

    # Adjust the layout
    plt.tight_layout()

    # Export the figure
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"CV Line plot saved as: {output_file}")
    plt.show()

    return fig


def plot_cv_radar_clean(cv_results, output_file='CV_Radar_Clean.png'):
    """
    Draw a clean radar chart with labels centered in each segment
    """
    # Prepare the input data
    wavelengths = cv_results['wavelength'].values
    n_segments = 13  # 13segments

    # Create 13 segments,one per 100 nm
    segment_ranges = [(400, 500), (500, 600), (600, 700),
                      (700, 800), (800, 900), (900, 1000),
                      (1000, 1100), (1100, 1200), (1200, 1300),
                      (1300, 1400), (1400, 1500), (1500, 1600),
                      (1600, 1700)]

    segment_labels = [f'{start}-{end}' for start, end in segment_ranges]

    # Calculate the mean CV for each segment
    segment_cv_c = []
    segment_cv_h = []

    for start_wl, end_wl in segment_ranges:
        mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)

        if mask.any():
            cv_c_segment = cv_results.loc[mask, 'cv_C'].mean()
            cv_h_segment = cv_results.loc[mask, 'cv_H'].mean()
        else:
            cv_c_segment = np.nan
            cv_h_segment = np.nan

        segment_cv_c.append(cv_c_segment)
        segment_cv_h.append(cv_h_segment)

    # Close the radar chart loop
    segment_cv_c_closed = segment_cv_c + [segment_cv_c[0]]
    segment_cv_h_closed = segment_cv_h + [segment_cv_h[0]]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(projection='polar'))

    # Configure angles
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    # Calculate the central angle of each segment
    angle_width = 2 * np.pi / n_segments
    center_angles = angles + angle_width / 2

    # Plot radar charts for groups C and H
    ax.plot(angles_closed, segment_cv_c_closed, 'o-', color='#1f77b4', linewidth=2.5,
            markersize=4, label='Control (C)', alpha=0.9)
    ax.fill(angles_closed, segment_cv_c_closed, color='#1f77b4', alpha=0.15)

    ax.plot(angles_closed, segment_cv_h_closed, 's-', color='#d62728', linewidth=2.5,
            markersize=4, label='Treatment (H)', alpha=0.9)
    ax.fill(angles_closed, segment_cv_h_closed, color='#d62728', alpha=0.15)

    # Hide default tick labels
    ax.set_xticks(angles)
    ax.set_xticklabels([''] * n_segments)

    # Apply consistent label formatting
    max_cv = max(max(segment_cv_c), max(segment_cv_h))
    label_radius = max_cv * 1.3  # Use a consistent radius

    for i, (angle, label) in enumerate(zip(center_angles, segment_labels)):
        # Convert radians to degrees
        angle_deg = np.degrees(angle)

        # Apply a uniform label rotation angle
        rotation_angle = angle_deg + 90

        # Align labels consistently
        ha = 'center'
        va = 'center'  # Use center alignment throughout

        # Draw labels
        ax.text(angle, label_radius, label,
                ha=ha, va=va,
                fontsize=11,
                fontweight='bold',
                color='black',
                rotation=rotation_angle,
                rotation_mode='anchor')

    # Configure the radial axis
    ax.set_ylim(0, max_cv * 1.4)
    ax.set_yticks(np.linspace(0, max_cv, 5))
    ax.set_yticklabels([f'{tick:.1f}%' for tick in np.linspace(0, max_cv, 5)],
                       fontsize=10, fontweight='bold')

    # Add a legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                       fontsize=14,
                       frameon=False)

    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Add a grid
    ax.grid(True, alpha=0.5, linestyle='-')

    # Adjust layout and Export
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Clean radar chart saved as: {output_file}")
    plt.show()

    return fig


def main():
    """
    Main function：Run the complete CV analysis workflow
    """
    # Set file paths
    reflectance_file = r"CH_A.xlsx"

    # Calculate the coefficient of variation
    print("=" * 60)
    print("Step1: Calculate the coefficient of variation")
    print("=" * 60)

    cv_results, cv_stats = calculate_coefficient_of_variation(
        reflectance_file,
        treatment_col='treatment'
    )


    # Plot radar chart
    print("\n" + "=" * 60)
    print("Step5: Plot radar chart")
    print("=" * 60)

    plot_cv_radar_clean(cv_results, output_file='Figure_2_CV_Radar_Chart.png')

    # Export results
    print("\n" + "=" * 60)
    print("Step6: Export results")
    print("=" * 60)

    cv_results = export_cv_results(cv_results, cv_stats)

   

if __name__ == "__main__":
    main()


##fig.2H
library(openxlsx)
library(sommer)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
data <- read.xlsx("CH_A.xlsx", 3)
kinship_ids <- read.table("kinship_matrix_square.rel.id", 
                          header = FALSE, stringsAsFactors = FALSE)$V1
K_matrix <- as.matrix(fread("kinship_matrix_square.rel", header = FALSE))
rownames(K_matrix) <- colnames(K_matrix) <- kinship_ids


material_ids <- as.character(unique(data[, 1]))
kinship_ids_char <- as.character(kinship_ids)
common_ids <- intersect(material_ids, kinship_ids_char)


data <- data[data[, 1] %in% common_ids, ]
K_used <- K_matrix[common_ids, common_ids]

# Separate data for treatments C and H
data_C <- data[data$Treatment == "C", ]
data_H <- data[data$Treatment == "H", ]


calculate_h2_single_env_v4 <- function(trait_values, genotype, K_mat, env_name) {
        
        df <- data.frame(
                trait = trait_values,
                genotype = as.character(genotype),
                stringsAsFactors = FALSE
        )
        
        df <- df[complete.cases(df), ]
        
        if(nrow(df) < 10 || var(df$trait, na.rm = TRUE) == 0) {
                return(list(H2 = NA, Vg = NA, Ve = NA, converged = FALSE))
        }
        
        tryCatch({
                df$genotype <- factor(df$genotype, levels = rownames(K_mat))
                
           
                model <- mmes(
                        fixed = trait ~ 1,
                        random = ~ vsm(genotype, Gu = K_mat),  
                        rcov = ~ units,
                        data = df,
                        verbose = FALSE,
                        date.warning = FALSE
                )
                
              
                var_comp <- summary(model)$varcomp
                
                Vg <- var_comp[grep("genotype", var_comp$VarComp), "VarCompEst"]
                Ve <- var_comp[grep("units", var_comp$VarComp), "VarCompEst"]
                
                if(length(Vg) == 0 || length(Ve) == 0) {
                        return(list(H2 = NA, Vg = NA, Ve = NA, converged = FALSE))
                }
                
                
                H2 <- Vg / (Vg + Ve)
                H2 <- max(0, min(1, H2))
                
                return(list(H2 = H2, Vg = Vg, Ve = Ve, converged = TRUE))
                
        }, error = function(e) {
                
                tryCatch({
                        library(lme4)
                        model_lme4 <- lmer(trait ~ 1 + (1|genotype), data = df)
                        var_comp <- as.data.frame(VarCorr(model_lme4))
                        
                        Vg <- var_comp$vcov[var_comp$grp == "genotype"]
                        Ve <- var_comp$vcov[var_comp$grp == "Residual"]
                        
                        H2 <- Vg / (Vg + Ve)
                        H2 <- max(0, min(1, H2))
                        
                        return(list(H2 = H2, Vg = Vg, Ve = Ve, converged = TRUE))
                        
                }, error = function(e2) {
                        return(list(H2 = NA, Vg = NA, Ve = NA, converged = FALSE))
                })
        })
}

calculate_h2_across_env_v4 <- function(trait_values, genotype, treatment, K_mat) {
        
        df <- data.frame(
                trait = trait_values,
                genotype = as.character(genotype),
                treatment = as.factor(treatment),
                stringsAsFactors = FALSE
        )
        
        df <- df[complete.cases(df), ]
        
        if(nrow(df) < 10 || var(df$trait, na.rm = TRUE) == 0) {
                return(list(H2 = NA, Vg = NA, Vge = NA, Ve = NA, converged = FALSE))
        }
        
        tryCatch({
                df$genotype <- factor(df$genotype, levels = rownames(K_mat))
                
                
                model <- mmes(
                        fixed = trait ~ treatment,
                        random = ~ vsm(genotype, Gu = K_mat) + 
                                vsm(genotype:treatment),  # G×E interaction
                        rcov = ~ units,
                        data = df,
                        verbose = FALSE,
                        date.warning = FALSE
                )
                
                var_comp <- summary(model)$varcomp
                
                Vg <- var_comp[grep("^genotype$", var_comp$VarComp), "VarCompEst"]
                Vge <- var_comp[grep("genotype:treatment", var_comp$VarComp), "VarCompEst"]
                Ve <- var_comp[grep("units", var_comp$VarComp), "VarCompEst"]
                
                if(length(Vg) == 0) Vg <- 0
                if(length(Vge) == 0) Vge <- 0
                if(length(Ve) == 0) return(list(H2 = NA, Vg = NA, Vge = NA, Ve = NA, converged = FALSE))
                
                
                n_env <- 2
                H2 <- Vg / (Vg + Vge/n_env + Ve/n_env)
                H2 <- max(0, min(1, H2))
                
                return(list(H2 = H2, Vg = Vg, Vge = Vge, Ve = Ve, converged = TRUE))
                
        }, error = function(e) {
                
                tryCatch({
                        library(lme4)
                        model_lme4 <- lmer(trait ~ treatment + (1|genotype), data = df)
                        var_comp <- as.data.frame(VarCorr(model_lme4))
                        
                        Vg <- var_comp$vcov[var_comp$grp == "genotype"]
                        Ve <- var_comp$vcov[var_comp$grp == "Residual"]
                        
                        n_env <- 2
                        H2 <- Vg / (Vg + Ve/n_env)
                        H2 <- max(0, min(1, H2))
                        
                        return(list(H2 = H2, Vg = Vg, Vge = 0, Ve = Ve, converged = TRUE))
                        
                }, error = function(e2) {
                        return(list(H2 = NA, Vg = NA, Vge = NA, Ve = NA, converged = FALSE))
                })
        })
}


pb <- txtProgressBar(min = 0, max = 464, style = 3)

results_list <- list()

for(i in 4:467) {
        band_idx <- i - 1
        
        
        trait_C <- data_C[, i]
        genotype_C <- data_C[, 1]
        result_C <- calculate_h2_single_env_v4(trait_C, genotype_C, K_used, "Control")
        
        
        trait_H <- data_H[, i]
        genotype_H <- data_H[, 1]
        result_H <- calculate_h2_single_env_v4(trait_H, genotype_H, K_used, "Drought")
        
        
        trait_all <- data[, i]
        genotype_all <- data[, 1]
        treatment_all <- data$Treatment
        result_across <- calculate_h2_across_env_v4(trait_all, genotype_all, treatment_all, K_used)
        
        results_list[[band_idx]] <- data.frame(
                Band_Number = band_idx,
                Band_Name = colnames(data)[i],
                
                H2_C = result_C$H2,
                Vg_C = result_C$Vg,
                Ve_C = result_C$Ve,
                
                H2_H = result_H$H2,
                Vg_H = result_H$Vg,
                Ve_H = result_H$Ve,
                
                H2_across = result_across$H2,
                Vg_across = result_across$Vg,
                Vge_across = result_across$Vge,
                Ve_across = result_across$Ve,
                
                Converged_C = result_C$converged,
                Converged_H = result_H$converged,
                Converged_across = result_across$converged,
                
                stringsAsFactors = FALSE
        )
        
        setTxtProgressBar(pb, band_idx)
}
close(pb)

heritability_df <- do.call(rbind, results_list)
write.csv(heritability_df, "heritability_complete_results.csv", row.names = FALSE)
write.xlsx(heritability_df, "heritability_complete_results.xlsx", row.names = FALSE)


h2_data <- heritability_df %>%
        dplyr::select(Band_Number, Band_Name, H2_across, Vg_across, Vge_across, Ve_across) %>%
        dplyr::filter(!is.na(H2_across))


library(ggplot2)

theme_pub <- theme_classic(base_size = 16) +
        theme(
                plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
                axis.title = element_text(face = "bold"),
                axis.text = element_text(color = "black",face = "bold"),
                panel.border = element_rect(color = "black", fill = NA, size = 1)
        )


p1 <- ggplot(h2_data, aes(x = Band_Number, y = H2_across)) +
        geom_line(color = "#2C3E50", size = 0.8) +
        geom_point(size = 0.8, alpha = 0.4, color = "#E74C3C") +
        geom_smooth(method = "loess", span = 0.15, 
                    color = "#3498DB", fill = "#3498DB", alpha = 0.15) +
        geom_hline(yintercept = mean(h2_data$H2_across), 
                   linetype = "dashed", color = "#E67E22") +
        labs(
                title = "Broad-sense Heritability across Spectral Bands",
                subtitle = "(Across Control and Drought Environments)",
                x = "Spectral Band Number",
                y = expression(paste("Heritability (", H^2, ")"))
        ) +
        scale_y_continuous(limits = c(0, 0.8), breaks = seq(0, 0.8, 0.1)) +
        theme_pub

h2_data$Wavelength <- as.numeric(h2_data$Band_Name)

p1 <- ggplot(h2_data, aes(x = Wavelength, y = H2_across)) +
        geom_line(color = "#1f77b4", size = 0.8, alpha = 0.8) +
        #geom_point(size = 0.8, alpha = 0.4, color = "#E74C3C") +
        geom_hline(yintercept = mean(h2_data$H2_across), 
                   linetype = "dotted", color = "#d62728", size = 0.8) +
        annotate("text", 
                 x = max(h2_data$Wavelength, na.rm = TRUE) * 0.85, 
                 y = mean(h2_data$H2_across) + 0.02,
                 label = paste0("Mean H² = ", 
                                round(mean(h2_data$H2_across), 3)),
                 color = "#d62728", size = 3.5, fontface = "bold") +
        labs(
                x = "Wavelength (nm)",
                y = expression(bold(paste("Heritability (", H^2, ")")))  # Use bold() inside expression()
        ) +
        scale_y_continuous(limits = c(0, max(h2_data$H2_across) * 1.1), 
                           breaks = seq(0, 0.8, 0.1)) +
        scale_x_continuous(breaks = seq(400, 1700, 100)) +
        theme_pub +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))


ggsave("Fig1_Heritability_Trend.pdf", p1, width = 12, height = 6)
ggsave("Fig1_Heritability_Trend1.png", p1, width = 6, height = 4, dpi = 300)

#Plot genome-wide profiles
# --------------------------
# Generate a PDF file
# --------------------------
pdf("ld_decay_merged_subABD_with_whole.pdf")

# 1. Define group colors
group_colors <- c("A" = "#009900", "B" = "#FF6600", "D" = "#0066FF")

# 2. Read subgenome data
ea <- read.table("ld_decay_merged_subABD.A") 
eb <- read.table("ld_decay_merged_subABD.B")  
ed <- read.table("ld_decay_merged_subABD.D")  

# 3. Read whole-genome data
ew <- read.table("ld_decay_merged_Whole.bin.gz")  # Whole-genome data file

# 4. Calculate the maximum y-axis value
max_r2 <- max(ea[,2], eb[,2], ed[,2], ew[,2], na.rm = TRUE)

# 5. Plot the base curve
plot(
  x = ea[,1]/1000,       # Convert distance to kb
  y = ea[,2],             # r²value
  type = "l",             # Line plot
  col = group_colors["A"],# Agroup color
  main = "LD decay",      
  xlab = "Distance(Kb)",  
  xlim = c(0, 20000),     
  ylim = c(0, max_r2),    
  ylab = expression(r^2), 
  bty = "n",              # No border
  lwd = 2                 # Line width
)

# 6. Overlay curves for groups B and D and the whole genome
lines(eb[,1]/1000, eb[,2], col = group_colors["B"], lwd = 2)  
lines(ed[,1]/1000, ed[,2], col = group_colors["D"], lwd = 2)  
lines(ew[,1]/1000, ew[,2], col = "black", lwd = 2)            # Whole genome (Black)

legend(
  "topright",                                             # Legend position
  legend = c("A", "B", "D", "Whole subgenome"),          # Legend labels
  col = c(group_colors["A"], group_colors["B"], group_colors["D"], "black"),  # Corresponding colors
  lty = 1,                                                # Solid line
  lwd = 2,                                                # Line width
  bty = "n",                                              # No border
  cex = 1                                                 # Font size
)

# Close PDF device
dev.off()


#fig4.B
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
import warnings

warnings.filterwarnings('ignore')

# Configure fonts and visualization settings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Specify global font sizes
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 15,
    'figure.titlesize': 14
})


def map_chromosome_name(chr_num):
    """Map numeric chromosome IDs to chromosome names"""
    if 1 <= chr_num <= 3:
        names = ['1A', '1B', '1D']
        return names[chr_num - 1]
    elif 4 <= chr_num <= 6:
        names = ['2A', '2B', '2D']
        return names[chr_num - 4]
    elif 7 <= chr_num <= 9:
        names = ['3A', '3B', '3D']
        return names[chr_num - 7]
    elif 10 <= chr_num <= 12:
        names = ['4A', '4B', '4D']
        return names[chr_num - 10]
    elif 13 <= chr_num <= 15:
        names = ['5A', '5B', '5D']
        return names[chr_num - 13]
    elif 16 <= chr_num <= 18:
        names = ['6A', '6B', '6D']
        return names[chr_num - 16]
    elif 19 <= chr_num <= 21:
        names = ['7A', '7B', '7D']
        return names[chr_num - 19]
    else:
        return f'Chr{chr_num}'


def process_qtl_data(file_path):
    """Process QTL data, add chromosome names, and count QTLs"""
    print("Processing QTL data...")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"Original data shape: {df.shape}")
    print(f"Data column names: {df.columns.tolist()}")

    df['Chromosome'] = pd.to_numeric(df['Chromosome'], errors='coerce')
    df['Chr_Name'] = df['Chromosome'].apply(map_chromosome_name)

    qtl_counts = {}
    for group in ['C', 'H']:
        group_data = df[df['Group'] == group]
        chr_counts = group_data['Chromosome'].value_counts().sort_index()
        for chr_num in range(1, 22):
            if chr_num not in chr_counts:
                chr_counts[chr_num] = 0
        qtl_counts[group] = chr_counts.sort_index()

    chr_names = [map_chromosome_name(i) for i in range(1, 22)]
    summary_df = pd.DataFrame({
        'Chromosome': list(range(1, 22)),
        'Chr_Name': chr_names,
        'C_Count': [qtl_counts['C'].get(i, 0) for i in range(1, 22)],
        'H_Count': [qtl_counts['H'].get(i, 0) for i in range(1, 22)],
        'Total_Count': [qtl_counts['C'].get(i, 0) + qtl_counts['H'].get(i, 0) for i in range(1, 22)]
    })

    print("\nQTL Summary statistics:")
    print("=" * 60)
    print(f"Total QTL count: {len(df)}")
    print("\nCounts by chromosome:")
    print(summary_df.to_string(index=False))

    return df, summary_df


def plot_circular_bar_chart(summary_df, output_file='Circular_Bar_Chart_QTL.png'):
    """Plot a circular bar chart"""
    print("\nPlotting circular bar chart...")

    chr_names = summary_df['Chr_Name'].values       # Chromosome name
    chr_nums = summary_df['Chromosome'].values      # Numeric chromosome ID
    c_counts = summary_df['C_Count'].values         # C group count
    h_counts = summary_df['H_Count'].values         # H group count

    color_c = '#1f77b4'
    color_h = '#d62728'

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

    n = len(chr_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # Chromosome reference angles
    width = 2 * np.pi / n * 0.35

    # Bar positions
    c_angles = angles - width / 2
    h_angles = angles + width / 2

    # Bar baseline
    bar_bottom = 0.2
    max_count = max(c_counts.max(), h_counts.max())
    c_heights = c_counts / max_count * 0.7
    h_heights = h_counts / max_count * 0.7

    # Plot bars for groups C and H
    bars_c = ax.bar(c_angles, c_heights, width=width, bottom=bar_bottom,
                    color=color_c, alpha=0.9, edgecolor='white', linewidth=1.5,
                    label='Control (C)', zorder=3)
    bars_h = ax.bar(h_angles, h_heights, width=width, bottom=bar_bottom,
                    color=color_h, alpha=0.9, edgecolor='white', linewidth=1.5,
                    label='Drought (H)', zorder=3)

    
    # --------------------------
    # Set radial label positions below the bar baseline
    label_radius = bar_bottom - 0.04

    for i in range(n):
        
        current_angle = angles[i]
        # Display labels vertically
        rotation = 0
        # Text alignment
        ha = 'center'
        va = 'center'

        ax.text(
            current_angle,
            label_radius,
            chr_names[i],
            ha=ha, va=va,
            fontsize=9, fontweight='bold',
            rotation=rotation,
            rotation_mode='anchor',
            color='black', zorder=4
        )
    # Value labels above bars
    for i, (c_count, h_count) in enumerate(zip(c_counts, h_counts)):
        if c_count > 0:
            c_label_radius = bar_bottom + c_heights[i] + 0.025
            ax.text(c_angles[i], c_label_radius, str(c_count),
                    ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=color_c)
        if h_count > 0:
            h_label_radius = bar_bottom + h_heights[i] + 0.025
            ax.text(h_angles[i], h_label_radius, str(h_count),
                    ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=color_h)

    # Refine polar chart styling
    ax.set_theta_offset(np.pi / 2)  # 0°Start at the top
    ax.set_theta_direction(-1)      # Clockwise direction
    ax.spines['polar'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(0, 0.9)
    ax.set_yticks([])
    ax.grid(False)

    # legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.7, 0.25),
              fontsize=15, frameon=False, fancybox=True,
              framealpha=0.9, edgecolor='black')

    # Central circle
    center_circle = plt.Circle((0, 0), 0.08, transform=ax.transData._b,
                               color='white', edgecolor='black', linewidth=2, zorder=5)
    ax.add_patch(center_circle)
 
    plt.subplots_adjust(left=0.02, right=0.975, top=0.98, bottom=0.03)
    # Export picture
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Circular bar chart saved as: {output_file}")
    plt.show()

    return fig

def main():
    qtl_file = r"QTL summary.xlsx"  

   

    df, summary_df = process_qtl_data(qtl_file)
    fig1 = plot_circular_bar_chart(summary_df, output_file='Figure_1_Circular_Bar_Chart_QTL.png')


if __name__ == "__main__":
    main()
	
	
#fig4.C
library(dplyr)
library(ggplot2)
library(openxlsx)
library(tidyr)

# ========== Import the data ==========

cat("========== Read QTL data ==========\n")

qtl_summary <- read.xlsx("QTL_Summary_with_Type.xlsx", sheet = 1)

# Parse the Trait_List column (semicolon-separated traits)
qtl_summary <- qtl_summary %>%
  mutate(
    # Count traits associated with each QTL
    N_Traits = sapply(strsplit(as.character(Trait_List), ";"), length)
  )

# Calculate trait-count frequencies

# Number of unique traits associated with each QTL
qtl_trait_count <- qtl_complete %>%
  group_by(Merged_QTL_ID, Treatment) %>%
  summarise(
    N_Traits = n_distinct(Filename),
    Trait_List = paste(unique(Filename), collapse = ";"),
    .groups = "drop"
  )

# Compute frequencies
trait_freq_complete <- qtl_trait_count %>%
  group_by(N_Traits) %>%
  summarise(
    Frequency = n(),
    .groups = "drop"
  ) %>%
  arrange(N_Traits)

# ========== Plotting: Bar chart grouped by QTL type ==========

# Define colors
type_colors <- c(
  "Common" = "#FF7F00",
  "C_Specific" = "#009900",
  "H_Specific" = "#0066FF"
)

p2 <- ggplot(trait_count_by_type, aes(x = factor(N_Traits), y = Frequency, fill = QTL_Type)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8, 
           color = "white", linewidth = 0.8) +
  geom_text(
    aes(label = Frequency), 
    position = position_dodge(width = 0.9),
    vjust = -0.5, size = 3.5, fontface = "bold"
  ) +
  scale_fill_manual(
    values = type_colors,
    labels = c("Common" = "Common QTL", 
               "C_Specific" = "C-Specific", 
               "H_Specific" = "H-Specific")
  ) +
  labs(
    x = "Number of Associated Traits",
    y = "Frequency (Number of QTLs)",
    title = "Distribution of Trait Count per QTL by Type",
    fill = "QTL Type"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11, color = "black",face = "bold"),
    axis.text.x = element_text(
      size = 11, 
      color = "black", 
      face = "bold",
      angle = 45,      # Rotation angle
      hjust = 1,       # Horizontal alignment
      vjust = 1        # Vertical alignment
    ),
    legend.position = "right",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11,face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", linewidth = 0.3),
    panel.border = element_rect(color = "black", linewidth = 1)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

ggsave("QTL_Trait_Count_by_Type.pdf", p2, width = 10, height = 5)

#fig4.D
options(stringsAsFactors = F)
library(tidyverse)
library(data.table)
library(ggpubr)
library(viridis)
setwd("gwas_result")
# Import the data
ped <- as.data.frame(fread("Myb10_selected_snp.ped", header = F, stringsAsFactors = F, sep = " "))
map <- read.table("Myb10_selected_snp.map", header = F)
pheno_data <- read.table("C_vis_dA215.txt", header = T)

# Process data
ped_SNP = data.frame(ped[1:6], mapply(paste0, ped[-c(1:6)][c(T,F)], ped[-c(1:6)][c(F,T)]))
colnames(ped_SNP)[7:length(ped_SNP)] = map$V2
colnames(pheno_data)[1] <- "V1"

all_plotData = merge(ped_SNP, pheno_data, by.x = "V1", by.y = "V1", all.x=TRUE)

plotData = all_plotData[,c("V1", "SNP-24520213", "C_vis_dA.215")]
colnames(plotData) = c("sample", "SNP", "value")
plotData = plotData[grep("AA|TT|CC|GG", plotData[,2]),]
plotData = plotData[which(plotData[,3] != -9),]
# Recode SNP genotypes
plotData$SNP <- recode(plotData$SNP, 
                       "AA" = "TT",
                       "TT" = "GG")

sample_size = plotData %>% group_by(SNP) %>% summarize(num=n())
plotData = plotData %>% left_join(sample_size) %>% mutate(group = paste0(SNP, "(", num, ")"))

# ================= Check SNP types =================
cat("SNP type counts:/n")
print(table(plotData$SNP))
cat("/n")

# ================= Calculate the y-axis range =================
y_min <- min(plotData$value, na.rm = TRUE)
y_max <- max(plotData$value, na.rm = TRUE)
y_range <- y_max - y_min

signif_space <- y_range * 0.15
y_expand_lower <- 0.05
y_expand_upper <- 0.25

y_lower <- y_min - y_range * y_expand_lower
y_upper <- y_max + y_range * y_expand_upper

signif_y_position <- y_max + y_range * 0.12


cat("  Data range:", round(y_min, 6), "to", round(y_max, 6), "/n")
cat("  Display range:", round(y_lower, 6), "to", round(y_upper, 6), "/n")
cat("  Significance annotation position:", round(signif_y_position, 6), "/n")

# ================= Configure custom colors =================
# Blue (control) and red (drought)
custom_colors <- c(
  "#1f77b4", # Blue - First SNP type
  "#d62728" # Red - Second SNP type
)

# ================= Plotting section =================
temp_plot = ggplot(plotData, aes(fill=group, y=value, x=group)) + 
  # Violin plot
  geom_violin(trim = TRUE, color = "gray", size = 0.8) +
  
  # Boxplot
  geom_boxplot(width = 0.2, 
               position = position_dodge(0.9),
               size = 0.8) +
  
  # Scatter points
  geom_point(position = position_jitterdodge(0.2), 
             size = 2,
             alpha = 0.6) +
  
  # Use custom colors
  scale_fill_manual(values = custom_colors) + 
  
  # # Set the Y-axis range
  scale_y_continuous(
    limits = c(y_lower, y_upper),
    expand = c(0, 0),
    breaks = scales::pretty_breaks(n = 6)
  ) +
  
  # Statistical test labels
  stat_compare_means(
    label = "p.signif", 
    method = "t.test", 
    size = 8,
    label.y = signif_y_position,
    label.x = 1.5,
    vjust = 0
  ) + 
  
  # X- and y-Axis labels
  xlab(expression(bold("SNP-24520213"))) + 
  ylab(expression(bold("C_vis_dA.215"))) +
  
  # Theme settings
  theme_bw(base_size = 18) +
  theme(
    text = element_text(color = "black", face = "bold", size = 18),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black", margin = margin(t = 10)),
    axis.title.y = element_text(size = 20, face = "bold", color = "black", margin = margin(r = 10)),
    axis.text.x = element_text(hjust = 0.5, vjust = 1, size = 16, color = "black", face = "bold"),
    axis.text.y = element_text(size = 16, color = "black", face = "bold"),
    axis.line = element_line(color = "black", size = 1),
    axis.ticks = element_line(color = "black", size = 0.8),
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(color = "black", size = 1.2, fill = NA),
    plot.margin = margin(t = 20, r = 10, b = 10, l = 10, unit = "pt")
  )

# Print plot preview
print(temp_plot)

# ================= Export the figure =================

# PDF format
ggsave("Myb10_haplotype_plot_custom_colors.pdf", 
       plot = temp_plot, 
       width = 8,
       height = 7,
       dpi = 300)


#fig4.E
#Prepare the input data for a manually constructed Myb10 network
library(dplyr)
library(openxlsx)

# Import the data
data <- read.xlsx("gwas_result/Myb10networkplotData.xlsx")
data$Pvalue <- as.numeric(data$Pvalue)

# Count occurrences of each gene_name/Filename pair
pair_counts <- data %>%
  group_by(gene_name, Filename) %>%
  summarise(
    Count = n(),
    .groups = 'drop'
  )

# Sum Count for each Filename (as a target-node attribute)
target_counts <- pair_counts %>%
  group_by(Filename) %>%
  summarise(
    Target_Count = sum(Count),
    .groups = 'drop'
  )

# Keep the minimum P-value for each pair
edges <- data %>%
  group_by(gene_name, Filename) %>%
  summarise(
    Pvalue = min(Pvalue),
    Count = n(),
    .groups = 'drop'
  ) %>%
  left_join(target_counts, by = "Filename")

# Inspect results
print(edges)

# Export the results
write.xlsx(edges, "gwas_result/processed_network_data.xlsx")


#fig.5A
library("CMplot")
library(openxlsx)
library(data.table)
library(dplyr)
library(ggplot2)

# Import the data
H <- read.xlsx("gwas_result/ddA276.xlsx", 2)
H$group <- "H"
C <- read.xlsx("gwas_result/ddA276.xlsx", 1)
C$group <- "C"

# Convert Pvalue to numeric before merging
H$Pvalue <- as.numeric(as.character(H$Pvalue))
C$Pvalue <- as.numeric(as.character(C$Pvalue))

# Remove rows with invalid P-values
H <- H[!is.na(H$Pvalue) & H$Pvalue > 0, ]
C <- C[!is.na(C$Pvalue) & C$Pvalue > 0, ]

HSI <- rbind(C, H)
pos <- read.table("position_add_03G.txt", header = T)

# Determine the maximum genomic position for each chromosome
chr_len <- pos[, c(3, 6)]

# Compute the starting offset for each chromosome
chr_pos <- pos[, c(3, 5)]

# Compute cumulative genomic positions for SNPs
chr_pos$Chromosome <- c(1:21)
Snp_pos <- HSI %>%
  left_join(chr_pos, by = "Chromosome") %>%
  mutate(BPcum = Position + START)

# X-axis ticks
X_axis <- pos[, c(3, 7)] 

# Configure mirrored plotting
Snp_pos$direction <- ifelse(Snp_pos$group == 'C', 1, -1)
Snp_pos$direction <- as.numeric(Snp_pos$direction)

# Calculate ypos
Snp_pos$ypos <- NA
valid_indices <- which(!is.na(Snp_pos$direction) & 
                         !is.na(Snp_pos$Pvalue) & 
                         Snp_pos$Pvalue > 0)

if(length(valid_indices) > 0) {
  Snp_pos$ypos[valid_indices] <- -log10(Snp_pos$Pvalue[valid_indices]) * Snp_pos$direction[valid_indices]
}

# Identify observations exceeding the upper threshold
thre1 <- Snp_pos %>% filter(!is.na(ypos) & abs(ypos) > -log10(4.03E-7))

# Threshold
threshold <- -log10(4.03E-7)

# ================= Plotting =================

p <- ggplot() +
  # Chromosome boundary lines
  geom_vline(xintercept = chr_len$END, 
             color = "gray70", 
             linetype = "dashed", 
             size = 0.5, 
             alpha = 0.7) +
  
  # Scatterplot - C group
  geom_point(data = Snp_pos[Snp_pos$group == "C" & !is.na(Snp_pos$ypos),], 
             aes(x = BPcum, y = ypos, color = as.factor(Chromosome)), 
             alpha = 0.8, 
             size = 1.2) +  
  
  # Scatterplot - H group
  geom_point(data = Snp_pos[Snp_pos$group == "H" & !is.na(Snp_pos$ypos),], 
             aes(x = BPcum, y = ypos, color = as.factor(Chromosome)), 
             alpha = 0.8, 
             size = 1.2) +
  
 
  scale_color_manual(values = rep(c("#009900", "#FF6600", "#0066FF"), 21)) +
  
  
  geom_hline(yintercept = 0, color = "white", size = 1.0) +
  geom_hline(yintercept = threshold, color = 'black', size = 0.8, linetype = "twodash") +
  geom_hline(yintercept = -threshold, color = 'black', size = 0.8, linetype = "twodash") +
  
  
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)), 
                     limits = c(-8, 8),
                     breaks = c(seq(-8, -4, 1), seq(4, 8, 1)),
                     labels = c(seq(8, 4, -1), seq(4, 8, 1))) +
  
  
  scale_x_continuous(expand = c(0.01, 0), 
                     labels = X_axis$Chr1, 
                     breaks = X_axis$x_lab) +
  
 
  labs(x = "Chromosome", 
       y = expression(bold(paste("-", log[10], "(P)")))) +
  
  # Theme settings
  theme_bw(base_size = 16) +  
  theme(
    # Global font settings: Black+Bold
    text = element_text(color = "black", face = "bold"),
    
    # Axis lines
    axis.line = element_line(color = "black", size = 0.8),
    axis.ticks = element_line(color = "black", size = 0.8),
    
    # Remove grid lines
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.border = element_rect(color = "black", size = 1.2),
    
    
    axis.text.x = element_text(
      angle = 45, 
      hjust = 1, 
      vjust = 1,
      size = 14,           # Increase font size
      color = "black",     # Black
      face = "bold"        # Bold
    ),
    
   
    axis.text.y = element_text(
      size = 14,           # Increase font size
      color = "black",     # Black
      face = "bold"        # Bold
    ),
    
    
    axis.title.x = element_text(
      size = 18,           # Increase font size
      face = "bold",       # Bold
      color = "black",     # Black
      margin = margin(t = 10)
    ),
    
    
    axis.title.y = element_text(
      size = 18,           # Increase font size
      face = "bold",       # Bold
      color = "black",     # Black
      margin = margin(r = 10)
    ),
    
    # Remove the legend
    legend.position = "none"
  )

# ================= Export the figure =================

# PNG format
ggsave(
  "ddA276_manhattan_plot_bold.png",
  plot = p,
  device = "png",
  width = 6,     # Increase width slightly
  height = 6,     # Increase height slightly
  dpi = 300,
  units = "in"
)

#fig.5B
#Heatmap + block + GWAS + gene
LDBlockShow -InVCF new498_sample_g25m5.vcf -OutPut G26_gwas -Region 5:150813676-151659794 -InGWAS H_vis_ddA.276 -OutPng -SeleVar 2 -InGFF gene_of_G26.gff 
ShowLDSVG -InPreFix G26_gwas -OutPut G26_gwas_plus -InGWAS H_vis_ddA.276 -Cutline 6.39 -InGFF gene_of_G26.gff -OutPng -SeleVar 2 -InGFF gene_of_G26.gff -SpeSNPName Spe.snp -ShowGWASSpeSNP

#fig.5C
#Haplotype plotting
options(stringsAsFactors = F)
library(tidyverse)
library(data.table)
library(ggpubr)
library(viridis)

# Import the data
ped <- as.data.frame(fread("selected_G26_related_snp.ped", header = F, stringsAsFactors = F, sep = " "))
map <- read.table("selected_G26_related_snp.map", header = F)
pheno_data <- read.table("H_vis_ddA.276.txt", header = T)

# Process data
ped_SNP = data.frame(ped[1:6], mapply(paste0, ped[-c(1:6)][c(T,F)], ped[-c(1:6)][c(F,T)]))
colnames(ped_SNP)[7:length(ped_SNP)] = map$V2
colnames(pheno_data)[1] <- "V1"

all_plotData = merge(ped_SNP, pheno_data, by.x = "V1", by.y = "V1", all.x=TRUE)

plotData = all_plotData[,c("V1", "SNP-12748536", "H_vis_ddA.276")]
colnames(plotData) = c("sample", "SNP", "value")
plotData = plotData[grep("AA|TT|CC|GG", plotData[,2]),]
plotData = plotData[which(plotData[,3] != -9),]

sample_size = plotData %>% group_by(SNP) %>% summarize(num=n())
plotData = plotData %>% left_join(sample_size) %>% mutate(group = paste0(SNP, "(", num, ")"))

# ================= Check SNP types =================
cat("SNP type counts:\n")
print(table(plotData$SNP))
cat("\n")

# ================= Calculate the y-axis range =================
y_min <- min(plotData$value, na.rm = TRUE)
y_max <- max(plotData$value, na.rm = TRUE)
y_range <- y_max - y_min

signif_space <- y_range * 0.15
y_expand_lower <- 0.05
y_expand_upper <- 0.25

y_lower <- y_min - y_range * y_expand_lower
y_upper <- y_max + y_range * y_expand_upper

signif_y_position <- y_max + y_range * 0.12


cat("  Data range:", round(y_min, 6), "to", round(y_max, 6), "\n")
cat("  Display range:", round(y_lower, 6), "to", round(y_upper, 6), "\n")
cat("  Significance annotation position:", round(signif_y_position, 6), "\n")

# ================= Configure custom colors =================
# Blue (control) and red (drought)
custom_colors <- c(
  "#1f77b4", # Blue - First SNP type
  "#d62728" # Red - Second SNP type
)

# ================= Plotting section =================
temp_plot = ggplot(plotData, aes(fill=group, y=value, x=group)) + 
  # Violin plot
  geom_violin(trim = TRUE, color = "gray", size = 0.8) +
  
  # Boxplot
  geom_boxplot(width = 0.2, 
               position = position_dodge(0.9),
               size = 0.8) +
  
  # Scatter points
  geom_point(position = position_jitterdodge(0.2), 
             size = 2,
             alpha = 0.6) +
  
  # Use custom colors
  scale_fill_manual(values = custom_colors) + 
  
  # Set the Y-axis range
  scale_y_continuous(
    limits = c(y_lower, y_upper),
    expand = c(0, 0),
    breaks = scales::pretty_breaks(n = 6)
  ) +
  
  # Statistical test labels
  stat_compare_means(
    label = "p.signif", 
    method = "t.test", 
    size = 8,
    label.y = signif_y_position,
    label.x = 1.5,
    vjust = 0
  ) + 
  
  # X- and y-Axis labels
  xlab(expression(bold("SNP-12748536"))) + 
  ylab(expression(bold("H_vis_ddA.276"))) +
  
  # Theme settings
  theme_bw(base_size = 18) +
  theme(
    text = element_text(color = "black", face = "bold", size = 18),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black", margin = margin(t = 10)),
    axis.title.y = element_text(size = 20, face = "bold", color = "black", margin = margin(r = 10)),
    axis.text.x = element_text(hjust = 0.5, vjust = 1, size = 16, color = "black", face = "bold"),
    axis.text.y = element_text(size = 16, color = "black", face = "bold"),
    axis.line = element_line(color = "black", size = 1),
    axis.ticks = element_line(color = "black", size = 0.8),
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(color = "black", size = 1.2, fill = NA),
    plot.margin = margin(t = 20, r = 10, b = 10, l = 10, unit = "pt")
  )

# Print plot preview
print(temp_plot)

# ================= Export the figure =================


# PNG format
ggsave("G26_haplotype_plot_custom_colors.png", 
       plot = temp_plot, 
       width = 3.6, 
       height = 3, 
       dpi = 600)

#fig.5F
library(ComplexHeatmap)
library(circlize)
library(openxlsx)
library(dplyr)

# ========== Import the data ==========
data <- read.xlsx("G26_mutunts_A.xlsx", sheet = 5)

# ========== Data preprocessing ==========
label_col <- data[[1]]  # Label
line_col <- data[[2]]   # Line
treatment_col <- data[[ncol(data)]]  # Treatment

# Extract reflectance data
band_cols <- data[, 3:(ncol(data)-1)]

# ========== # Ensure all columns are numeric ==========
band_cols <- as.data.frame(lapply(band_cols, as.numeric))

# Convert to a matrix
band_matrix <- as.matrix(band_cols)

# Handle NA values
if(any(is.na(band_matrix))) {
  
  for(j in 1:ncol(band_matrix)) {
    band_matrix[is.na(band_matrix[, j]), j] <- mean(band_matrix[, j], na.rm = TRUE)
  }
}

rownames(band_matrix) <- line_col


# ========== Data standardization ==========

band_scaled <- scale(band_matrix, center = TRUE, scale = TRUE)
cat("Standardization completed\n")

# ========== Band clustering ==========

set.seed(123)
n_clusters <- 4

km_result <- kmeans(t(band_scaled), centers = n_clusters, nstart = 25, iter.max = 100)
column_km <- km_result$cluster


print(table(column_km))

# ========== Extract wavelengths (remove the 'X' prefix) ==========

band_names <- colnames(band_cols)
wavelengths <- as.numeric(gsub("^X", "", band_names))


print(head(wavelengths))
cat("Wavelength range:", round(min(wavelengths, na.rm = TRUE), 1), "-", 
    round(max(wavelengths, na.rm = TRUE), 1), "nm\n")

# Check if any NA values remain
if(any(is.na(wavelengths))) {
  cat("Warning: There are still", sum(is.na(wavelengths)), "wavelengths that could not be parsed\n")
  cat("Problematic column names:\n")
  print(band_names[is.na(wavelengths)])
}

# ========== # Calculate clustering information and sort by wavelength ==========

cluster_info <- data.frame()

for(i in 1:n_clusters) {
  cluster_indices <- which(column_km == i)
  cluster_wavelengths <- wavelengths[cluster_indices]
  
  info <- data.frame(
    Cluster_ID = i,
    N_Bands = length(cluster_wavelengths),
    Min_Wavelength = round(min(cluster_wavelengths, na.rm = TRUE), 1),
    Max_Wavelength = round(max(cluster_wavelengths, na.rm = TRUE), 1),
    Mean_Wavelength = round(mean(cluster_wavelengths, na.rm = TRUE), 1)
  )
  
  cluster_info <- rbind(cluster_info, info)
}

cluster_info <- cluster_info[order(cluster_info$Mean_Wavelength), ]

print(cluster_info)

desired_order <- cluster_info$Cluster_ID
order_bands <- order(match(column_km, desired_order))
column_km_ordered <- column_km[order_bands]


# Specify the material order
label_display_order <- c("BW", "bb", "dd", "bbdd")
treatment_order <- c("C", "H")

all_combinations <- expand.grid(
  Label = label_display_order,
  Treatment = treatment_order,
  stringsAsFactors = FALSE
)

all_combinations <- all_combinations %>%
  mutate(Group = paste(Label, Treatment, sep = "-"))

# Create a matrix
n_groups <- nrow(all_combinations)
label_matrix <- matrix(NA, nrow = n_groups, ncol = ncol(band_scaled))
rownames(label_matrix) <- all_combinations$Group
colnames(label_matrix) <- colnames(band_scaled)

# Populate the matrix
for(i in 1:n_groups) {
  target_label <- all_combinations$Label[i]
  target_treatment <- all_combinations$Treatment[i]
  
  matching_indices <- which(label_col == target_label & treatment_col == target_treatment)
  
  if(length(matching_indices) > 0) {
    label_matrix[i, ] <- colMeans(band_scaled[matching_indices, , drop = FALSE])
    cat("Found", target_label, "-", target_treatment, ":", length(matching_indices), "samples\n")
  } else {
    cat("Warning: No data found for", target_label, "-", target_treatment)
  }
}

# ========== Color Mapping ==========
col_fun <- colorRamp2(
  c(min(label_matrix, na.rm = TRUE), 0, max(label_matrix, na.rm = TRUE)), 
  c("#0571B0", "white", "#CA0020")
)

# ========== Create Cluster Labels (Display Wavelength Ranges) ==========

cluster_labels <- sapply(1:n_clusters, function(i) {
  cluster_id <- desired_order[i]
  info <- cluster_info[cluster_info$Cluster_ID == cluster_id, ]
  
  min_wl <- round(info$Min_Wavelength, 1)
  max_wl <- round(info$Max_Wavelength, 1)
  
  sprintf("Cluster %d\n(%.1f-%.1f nm)", i, min_wl, max_wl)
})

print(cluster_labels)

# ========== Generate the heatmap ==========

ht <- Heatmap(
  label_matrix[, order_bands],
  name = "Standardized\nReflectance",
  col = col_fun,
  
  # Row settings
  cluster_rows = FALSE,
  show_row_names = TRUE,
  row_names_side = "left",
  row_names_gp = gpar(fontsize = 14, fontface = "bold", color = "black"),
  row_title = "Sample Group",
  row_title_gp = gpar(fontsize = 16, fontface = "bold", color = "black"),
  
  # Column settings
  cluster_columns = FALSE,
  show_column_names = FALSE,
  column_split = factor(column_km_ordered, levels = unique(column_km_ordered)),
  column_title = cluster_labels,
  column_title_gp = gpar(fontsize = 6, fontface = "bold", color = "black"),
  column_gap = unit(2, "mm"),
  
  # Border
  border = FALSE,
  
  # Legend
  heatmap_legend_param = list(
    title = "Standardized\nReflectance",
    title_gp = gpar(fontsize = 14, fontface = "bold", color = "black"),
    labels_gp = gpar(fontsize = 12, color = "black"),
    direction = "vertical",
    border = "black",
    legend_height = unit(6, "cm")
  )
)
ht
# ========== Export heatmap ==========
pdf("G26_spectral_heatmap_wavelength_4materials.pdf", width = 14, height = 6)
draw(ht, 
     column_title = "Spectral Bands Grouped by Wavelength",
     column_title_gp = gpar(fontsize = 16, fontface = "bold", color = "black"),
     heatmap_legend_side = "right",
     padding = unit(c(2, 2, 2, 10), "mm"))
dev.off()


# # Export clustering information
cluster_summary <- data.frame(
  Wavelength = wavelengths,
  Original_Cluster = column_km,
  Display_Order = match(column_km, desired_order)
) %>%
  arrange(Display_Order, Wavelength)

write.csv(cluster_summary, "band_cluster_wavelength_info.csv", row.names = FALSE)
write.xlsx(cluster_summary, "band_cluster_wavelength_info.xlsx")

#fig.5G
library(ggplot2)
library(dplyr)
library(tidyr)
library(openxlsx)

# Import the data
data <- read.xlsx("G26_mutants_A.xlsx", sheet = 1)
# Inspect the data structure
print(colnames(data)[1:10])
# ================= Data Preprocessing =================

# Extract Column names
label_col <- data[[1]]        # Label 
line_col <- data[[2]]         # Line 
treatment_col <- data[[ncol(data)]]  # Treatment (C/H)

# Extract reflectance data
band_cols <- data[, 3:(ncol(data)-1)]

# Convert to numeric
band_cols <- as.data.frame(lapply(band_cols, as.numeric))

# Extract wavelengths (remove the 'X' prefix from column names)
band_names <- colnames(band_cols)

# Remove the "X" prefix
wavelengths <- as.numeric(gsub("^X", "", band_names))

# Check for missing values
if(any(is.na(wavelengths))) {
  cat("Warning:", sum(is.na(wavelengths)), "wavelengths could not be parsed\n")
  cat("Problematic column names:\n")
  print(band_names[is.na(wavelengths)])
}
  
  # Remove columns with missing wavelength values
  valid_idx <- !is.na(wavelengths)
  band_cols <- band_cols[, valid_idx]
  wavelengths <- wavelengths[valid_idx]
  band_names <- band_names[valid_idx]
}

cat("Number of valid bands:", length(wavelengths), "\n")
cat("Wavelength range:", round(min(wavelengths, na.rm = TRUE), 1), "-",
    round(max(wavelengths, na.rm = TRUE), 1), "nm\n")

# ================= Calculate the Mean and Standard Error for Each Material-Treatment Combination =================

# Create data in long format
data_long <- data.frame(
  Label = label_col,
  Line = line_col,
  Treatment = treatment_col,
  band_cols
)

# Convert to long format (one row per wavelength)
data_long <- data_long %>%
  pivot_longer(
    cols = starts_with("X"),  
    names_to = "Wavelength_raw",
    values_to = "Reflectance"
  ) %>%
  mutate(
    Wavelength = as.numeric(gsub("^X", "", Wavelength_raw))  
  )

# Remove rows with NA wavelengths
data_long <- data_long %>%
  filter(!is.na(Wavelength))

# Compute summary statistics
stat_data <- data_long %>%
  group_by(Label, Treatment, Wavelength) %>%
  summarise(
    n = n(),
    mean_reflectance = mean(Reflectance, na.rm = TRUE),
    sd_reflectance = sd(Reflectance, na.rm = TRUE),
    se_reflectance = sd(Reflectance, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

# ========== Set the factor levels for the four materials ==========
stat_data$Label <- factor(stat_data$Label, levels = c("BW", "bb", "dd", "bbdd"))

# Create a combined grouping variable
stat_data$Group <- paste(stat_data$Label, stat_data$Treatment, sep = "-")

# ========== Set the factor levels for the eight material-treatment combinations ==========
stat_data$Group <- factor(stat_data$Group, 
                          levels = c("BW-C", "BW-H", "bb-C", "bb-H", 
                                     "dd-C", "dd-H", "bbdd-C", "bbdd-H"))


# ========== Define 8 Colors (4 Materials × 2 Treatments) ==========
combined_colors <- c(
  "BW-C" = "#A50026", 
  "bb-C" = "#D73027", 
  "dd-C" = "#F46D43", 
  "bbdd-C" = "#FDAE61", 
  "BW-H" = "#313695", 
  "bb-H" = "#4575B4", 
  "dd-H" = "#74ADD1", 
  "bbdd-H" = "#ABD9E9" 
)

# Custom plotting theme
custom_theme <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(size = 14, face = "bold", color = "black"),
    axis.title.y = element_text(size = 14, face = "bold", color = "black"),
    axis.text.x = element_text(size = 12, color = "black", face = "bold", 
                               angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 12, color = "black", face = "bold"),
    
    # Legend settings
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_text(size = 13, face = "bold", color = "black"),
    legend.text = element_text(size = 11, color = "black", face = "bold"),
    legend.box = "horizontal",
    legend.margin = margin(b = 10),
    legend.key.height = unit(0.25, "cm"),
    legend.key.width = unit(0.5, "cm"),
    
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )


# ========== Plotting ==========

p_all <- ggplot(stat_data, aes(x = Wavelength, y = mean_reflectance, color = Group, fill = Group)) +
  
geom_line(
  linewidth = 0.6
) +
  
  # ========== setting color ==========
scale_color_manual(
  values = combined_colors, 
  name = "Material",
  labels = c("BW-C" = "BW_C",
             "BW-H" = "BW_H",
             "bb-C" = "bb_C",
             "bb-H" = "bb_H",
             "dd-C" = "dd_C",
             "dd-H" = "dd_H",
             "bbdd-C" = "bbdd_C",
             "bbdd-H" = "bbdd_H")
) +
  scale_fill_manual(
    values = combined_colors, 
    name = "Material",
    labels = c("BW-C" = "BW_C",
               "BW-H" = "BW_H",
               "bb-C" = "bb_C",
               "bb-H" = "bb_H",
               "dd-C" = "dd_C",
               "dd-H" = "dd_H",
               "bbdd-C" = "bbdd_C",
               "bbdd-H" = "bbdd_H")
  ) +
  
  # X-axis settings
  scale_x_continuous(
    breaks = seq(400, 1000, by = 100),
    minor_breaks = seq(400, 1000, by = 50)
  ) +
  
  # Axis labels
  labs(
    x = "Wavelength (nm)",
    y = "Reflectance"
  ) +
  
  # theme
  custom_theme

print(p_all)

# ================= Export the figure =================

# PDF format
ggsave(
  filename = "Spectral_Reflectance_VIS_Combined_4materials.pdf",
  plot = p_all,
  width = 7,
  height = 3
)

#fig.5H
library(ggplot2)
library(dplyr)
library(tidyr)
library(openxlsx)

# Import the data
data <- read.xlsx("G26_mutants_A.xlsx", sheet = 2)

# Inspect the data structure
print(colnames(data)[1:10])

# ================= Data Preprocessing =================
# Extract Column names
label_col <- data[[1]]        
line_col <- data[[2]]        
treatment_col <- data[[ncol(data)]]  # Treatment (C/H)

# Extract reflectance data
band_cols <- data[, 3:(ncol(data)-1)]

# Convert to numeric
band_cols <- as.data.frame(lapply(band_cols, as.numeric))

# Extract wavelengths (remove the 'X' prefix from column names)
band_names <- colnames(band_cols)

# Remove the "X" prefix
wavelengths <- as.numeric(gsub("^X", "", band_names))

# Check for missing values
if(any(is.na(wavelengths))) {
  cat("Warning：have", sum(is.na(wavelengths)), "itemswavelengthcould not parse\n")
  cat("ProblematicColumn names:\n")
  print(band_names[is.na(wavelengths)])
  
  # Remove columns with missing wavelength values
  valid_idx <- !is.na(wavelengths)
  band_cols <- band_cols[, valid_idx]
  wavelengths <- wavelengths[valid_idx]
  band_names <- band_names[valid_idx]
}

cat("Number of valid bands:", length(wavelengths), "\n")
cat("Wavelength range:", round(min(wavelengths, na.rm = TRUE), 1), "-",
    round(max(wavelengths, na.rm = TRUE), 1), "nm\n")

# ================= Calculate the Mean and Standard Error for Each Material-Treatment Combination =================

# Create data in long format
data_long <- data.frame(
  Label = label_col,
  Line = line_col,
  Treatment = treatment_col,
  band_cols
)

# Convert to long format (one row per wavelength)
data_long <- data_long %>%
  pivot_longer(
    cols = starts_with("X"), # Select all columns starting with "X" 
    names_to = "Wavelength_raw",
    values_to = "Reflectance"
  ) %>%
  mutate(
    Wavelength = as.numeric(gsub("^X", "", Wavelength_raw))  # Remove the "X" prefix 
  )

# Remove rows with NA wavelengths
data_long <- data_long %>%
  filter(!is.na(Wavelength))

# Compute summary statistics
stat_data <- data_long %>%
  group_by(Label, Treatment, Wavelength) %>%
  summarise(
    n = n(),
    mean_reflectance = mean(Reflectance, na.rm = TRUE),
    sd_reflectance = sd(Reflectance, na.rm = TRUE),
    se_reflectance = sd(Reflectance, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

# Set the factor levels for the four materials 
stat_data$Label <- factor(stat_data$Label, levels = c("BW", "bb", "dd", "bbdd"))

# Create a combined grouping variable
stat_data$Group <- paste(stat_data$Label, stat_data$Treatment, sep = "-")

# Set the factor levels for the eight material-treatment combinations
stat_data$Group <- factor(stat_data$Group, 
                          levels = c("BW-C", "BW-H", "bb-C", "bb-H", 
                                     "dd-C", "dd-H", "bbdd-C", "bbdd-H"))

# ================= Define Colors =================
# Define 8 Colors (4 Materials × 2 Treatments)
combined_colors <- c(
  "BW-C" = "#A50026", 
  "bb-C" = "#D73027", 
  "dd-C" = "#F46D43", 
  "bbdd-C" = "#FDAE61", 
  "BW-H" = "#313695", 
  "bb-H" = "#4575B4", 
  "dd-H" = "#74ADD1", 
  "bbdd-H" = "#ABD9E9" 
)

# ================= Custom plotting theme =================

custom_theme <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(size = 14, face = "bold", color = "black"),
    axis.title.y = element_text(size = 14, face = "bold", color = "black"),
    axis.text.x = element_text(size = 12, color = "black", face = "bold", 
                               angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 12, color = "black", face = "bold"),
    
    # legend settings
    legend.position = "top",              
    legend.direction = "horizontal",      
    legend.title = element_text(size = 13, face = "bold", color = "black"),
    legend.text = element_text(size = 11, color = "black", face = "bold"),
    legend.box = "horizontal",            
    legend.margin = margin(b = 10),       
    legend.key.height = unit(0.25, "cm"), 
    legend.key.width = unit(0.5, "cm"),   
    
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ================= Plotting =================

p_all <- ggplot(stat_data, aes(x = Wavelength, y = mean_reflectance, color = Group, fill = Group)) +
    geom_line(
    linewidth = 0.3
  ) +
  
  # color settings 
  scale_color_manual(
    values = combined_colors, 
    name = "Material",
    labels = c("BW-C" = "BW_C",
               "BW-H" = "BW_H",
               "bb-C" = "bb_C",
               "bb-H" = "bb_H",
               "dd-C" = "dd_C",
               "dd-H" = "dd_H",
               "bbdd-C" = "bbdd_C",
               "bbdd-H" = "bbdd_H")
  ) +
  scale_fill_manual(
    values = combined_colors, 
    name = "Material",
    labels = c("BW-C" = "BW_C",
               "BW-H" = "BW_H",
               "bb-C" = "bb_C",
               "bb-H" = "bb_H",
               "dd-C" = "dd_C",
               "dd-H" = "dd_H",
               "bbdd-C" = "bbdd_C",
               "bbdd-H" = "bbdd_H")
  ) +
  
  # X-axis settings
  scale_x_continuous(
    breaks = seq(900, 1700, by = 100),
    minor_breaks = seq(900, 1700, by = 50)
  ) +
  
  # Axis labels
  labs(
    x = "Wavelength (nm)",
    y = "Reflectance"
  ) +
  
  # Theme
  custom_theme

print(p_all)

# ================= Export the figure =================

# PNG format
ggsave(
  filename = "Spectral_Reflectance_NIR_Combined.png",
  plot = p_all,
  width = 5,      
  height = 4,
  dpi = 300
)

# PDF format
ggsave(
  filename = "Spectral_Reflectance_All_Combined.pdf",
  plot = p_all,
  width = 7,      
  height = 3
)




#fig.5I
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 6)
data <- data[-1]
head(data)

# Specify the material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$ddA276 <- as.numeric(data$ddA276)


data$ddA276_abs <- abs(data$ddA276)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(ddA276),
    sd = sd(ddA276),
    se = sd(ddA276)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# T-test function
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$ddA276[treatment_data$Material == "BW"]
  bb_data <- treatment_data$ddA276[treatment_data$Material == "bb"]
  dd_data <- treatment_data$ddA276[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$ddA276[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== T-test Results ==========\n")
print(ttest_results)

# ========== Prepare Text for Significance Markers ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# Define 4 Colors
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = ddA276_abs, fill = Material)) +
  
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color = "black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # setting colors
  scale_fill_manual(values = custom_colors) +
 
scale_y_continuous(
  labels = function(x) paste0("-", format(x, scientific = TRUE)),
  limits = c(5e-05, 10e-05),  
  expand = c(0, 0)  
) +
  
  # siginificant maker
  geom_signif(
    xmin = c(0.7, 0.7, 0.7, 1.7, 1.7, 1.7),
    xmax = c(0.9, 1.1, 1.3, 1.9, 2.1, 2.3),
    annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                    annot_H_bb, annot_H_dd, annot_H_bbdd),
    y_position = c(8.8e-05, 9.2e-05, 9.6e-05, 8.8e-05, 9.2e-05, 9.6e-05),
    textsize = 7,
    vjust = 0.1,
    color = "black",
    tip_length = 0,
    parse = TRUE
  ) +
  
  labs(
    x = "Treatment",
    y = "ddA276 Value"
  ) +
  custom_theme_advanced

# Export the figure
ggsave("G26_ddA276_boxplot_4materials.png", p, width = 6.5, height = 4, dpi = 300)
ggsave("G26_ddA276_boxplot_4materials.pdf", p, width = 8, height = 7)
ggsave("G26_ddA276_boxplot_4materials.tiff", p, width = 8, height = 7, dpi = 600, compression = "lzw")

#fig.5J
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 1)
data <- data[-1]
head(data)

# Specify the material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$TKW <- as.numeric(data$TKW)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(TKW),
    sd = sd(TKW),
    se = sd(TKW)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# T-test function
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$TKW[treatment_data$Material == "BW"]
  bb_data <- treatment_data$TKW[treatment_data$Material == "bb"]
  dd_data <- treatment_data$TKW[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$TKW[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== T-test Results ==========\n")
print(ttest_results)

# ========== Prepare Text for Significance Markers ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')") 
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# Define 4 Colors
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = TKW, fill = Material)) +
  
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color = "black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # setting colors
  scale_fill_manual(values = custom_colors) +
 
  # ========== Statistical significance annotations==========
geom_signif(
  xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
  xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
  annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                  annot_H_bb, annot_H_dd, annot_H_bbdd),
  y_position = c(38, 39, 40, 38, 39, 40),
  textsize = 7,
  vjust = 0.1,
  color = "black",
  tip_length = 0,
  parse = TRUE
) +
  
  labs(
    x = "Treatment",
    y = "TKW (g)"
  )

# # Apply the theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(28, 42))


# Export the figure
ggsave("mutant_TKW_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)
ggsave("mutant_TKW_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_TKW_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")

#fig.5K
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 4)
data <- data[-1]
head(data)

# Specify the material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Grain_Length <- as.numeric(data$Grain_Length)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(Grain_Length),
    sd = sd(Grain_Length),
    se = sd(Grain_Length)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# T-test function
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$Grain_Length[treatment_data$Material == "BW"]
  bb_data <- treatment_data$Grain_Length[treatment_data$Material == "bb"]
  dd_data <- treatment_data$Grain_Length[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$Grain_Length[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== T-test Results ==========\n")
print(ttest_results)

# ========== Prepare Text for Significance Markers ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# Define 4 Colors
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Grain_Length, fill = Material)) +
  
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +

  geom_point(
    color = "black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # setting colors
  scale_fill_manual(values = custom_colors) +
  
  # siginificant maker
  geom_signif(
    xmin = c(0.7, 0.7, 0.7, 1.7, 1.7, 1.7),
    xmax = c(0.9, 1.1, 1.3, 1.9, 2.1, 2.3),
    annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                    annot_H_bb, annot_H_dd, annot_H_bbdd),
    y_position = c(6.0, 6.1, 6.2, 6.0, 6.1, 6.2),
    textsize = 7,
    vjust = 0.1,
    color = "black",
    tip_length = 0,
    parse = TRUE
  ) +
  
  labs(
    x = "Treatment",
    y = "Grain Length (mm)"
  )

# Apply the theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(5.5, 6.3))


# Export the figure
ggsave("mutant_Grain_Length_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)
ggsave("mutant_Grain_Length_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_Grain_Length_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")


#fig.5L

library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 5)
data <- data[-1]
head(data)

# Specify the material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Grain_Width <- as.numeric(data$Grain_Width)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(Grain_Width, na.rm = TRUE),
    sd = sd(Grain_Width, na.rm = TRUE),
    se = sd(Grain_Width, na.rm = TRUE)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# T-test function
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$Grain_Width[treatment_data$Material == "BW"]
  bb_data <- treatment_data$Grain_Width[treatment_data$Material == "bb"]
  dd_data <- treatment_data$Grain_Width[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$Grain_Width[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n==========  T-test Results ==========\n")
print(ttest_results)

# ========== Prepare Text for Significance Markers ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# Define 4 Colors
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Grain_Width, fill = Material)) +
  
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color = "black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # setting colors
  scale_fill_manual(values = custom_colors) +
  
  # Significance markers 
  geom_signif(
    xmin = c(0.7, 0.7, 0.7, 1.7, 1.7, 1.7),
    xmax = c(0.9, 1.1, 1.3, 1.9, 2.1, 2.3),
    annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                    annot_H_bb, annot_H_dd, annot_H_bbdd),
    y_position = c(3.4, 3.45, 3.5, 3.4, 3.45, 3.5),
    textsize = 7,
    vjust = 0.1,
    color = "black",
    tip_length = 0,
    parse = TRUE
  ) +
  
  labs(
    x = "Treatment",
    y = "Grain Width (mm)"
  )

# Apply the theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(2.9, 3.6))


# Export the figure
ggsave("mutant_Grain_Width_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)
ggsave("mutant_Grain_Width_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_Grain_Width_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")

#fig.5M
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)

data <- read.xlsx("ddA276_for_plot.xlsx", 8)
data <- data[-1]
head(data)

# ==========Change material order to 4 items ==========
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Protein <- as.numeric(data$Protein)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(Protein),
    sd = sd(Protein),
    se = sd(Protein)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ========== ttest ==========
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$Protein[treatment_data$Material == "BW"]
  bb_data <- treatment_data$Protein[treatment_data$Material == "bb"]
  dd_data <- treatment_data$Protein[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$Protein[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ========== Define 4 types of color ==========
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ================= Plotting section =================
p <- ggplot(data, aes(x = Treatment, y = Protein, fill = Material)) +
  
  # Boxplot
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color ="black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # Color settings
  scale_fill_manual(values = custom_colors) +
  
  # ========== Statistical significance annotations ==========
geom_signif(
  xmin = c(0.7, 0.7, 0.7, 1.7, 1.7, 1.7),
  xmax = c(0.9, 1.1, 1.3, 1.9, 2.1, 2.3),
  annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                  annot_H_bb, annot_H_dd, annot_H_bbdd),
  y_position = c(13.5, 14, 14.5, 14, 14.5, 15),
  textsize = 7,
  vjust = 0.1,
  color = "black",
  tip_length = 0,
  parse = TRUE
) +
  
  # Axis labels
  labs(
    x = "Treatment",
    y = "Protein Content (%)"
  )

# Apply the plotting theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(11, 16))

# Display plot
print(b)

# ================= Export the figure =================

ggsave("mutant_Protein_content_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)
ggsave("mutant_Protein_content_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_Protein_content_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")

#fig.5N
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)

setwd("2020_raw phenotype data")
data <- read.xlsx("ddA276_for_plot.xlsx", 3)
data <- data[-1]
head(data)

# ========== Set material order ==========
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Carbon <- as.numeric(data$Carbon)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(Carbon),
    sd = sd(Carbon),
    se = sd(Carbon)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ========== ttest==========
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$Carbon[treatment_data$Material == "BW"]
  bb_data <- treatment_data$Carbon[treatment_data$Material == "bb"]
  dd_data <- treatment_data$Carbon[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$Carbon[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ========== Define 4 types of color ==========
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ================= Plotting section =================

p <- ggplot(data, aes(x = Treatment, y = Carbon, fill = Material)) +
  
  # Boxplot
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color ="black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  
  # Color settings
  scale_fill_manual(values = custom_colors) +
  
  # ========== Statistical significance annotations ==========
geom_signif(
  xmin = c(0.7, 0.7, 0.7, 1.7, 1.7, 1.7),
  xmax = c(0.9, 1.1, 1.3, 1.9, 2.1, 2.3),
  annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                  annot_H_bb, annot_H_dd, annot_H_bbdd),
  y_position = c(44.5, 45, 45.5, 44.5, 45, 45.5),
  textsize = 7,
  vjust = 0.1,
  color = "black",
  tip_length = 0,
  parse = TRUE
) +
  
  # Axis labels
  labs(
    x = "Treatment",
    y = "Carbon Content (%)"
  )

# Apply the plotting theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(42.5, 46))

# Display plot
print(b)

# ================= Export the figure =================

ggsave("mutant_Carbon_content_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)
ggsave("mutant_Carbon_content_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_Carbon_content_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")

#fig.5O

install.packages(c("ggplot2", "readr", "dplyr"))

library(ggplot2)
library(openxlsx)
library(dplyr)

setwd("20260520")
# Import the data from both samples
BW <- read.xlsx("01.xlsx",2)
bbdd <- read.xlsx("03.xlsx",2)

# Organize Column names and add sample names
BW <- BW %>%
        rename(Position = `Position(mm)`) %>%
        mutate(Sample = "BW")

bbdd <- bbdd %>%
        rename(Position = `Position(mm)`) %>%
        mutate(Sample = "bbdd")

# Merge data
df <- bind_rows(BW, bbdd)

# Specify the sample order
df$Sample <- factor(df$Sample, levels = c("BW", "bbdd"))

# Make bbdd shift right as a whole
shift_value <- 5   

df <- df %>%
        mutate(
                Position_plot = ifelse(Sample == "bbdd",
                                       Position + shift_value,
                                       Position)
        )

# Plot
p <- ggplot(df, aes(x = Position_plot, y = Absorbance, color = Sample)) +
        geom_point(size = 0.2, alpha = 0.8) +
        #geom_line(linewidth = 0.8, alpha = 0.8) +
        scale_color_manual(
                values = c("BW" = "#0072B2", "bbdd" = "#D55E00")
        ) +
        labs(
                x = "Position (mm)",
                y = "Absorbance",
                color = "Sample"
        ) +
        theme_classic(base_size = 14) +
        theme(
                axis.text = element_text(color = "black"),
                axis.title = element_text(color = "black", face = "bold"),
                legend.title = element_text(face = "bold")
        )

p

# Export
ggsave("Absorbance_H_plot.png", p, width = 6, height = 4, dpi = 300)
ggsave("Absorbance_shifted_plot.pdf", p, width = 6, height = 4)

#fig.5P

library(ggplot2)
library(openxlsx)
library(dplyr)

setwd("D:/sucrosedensity gradient separation/20260601")
# Import the data from both samples
BW <- read.xlsx("07.xlsx",2)
bbdd <- read.xlsx("06.xlsx",2)

# Organize Column names and add sample names
BW <- BW %>%
  rename(Position = `Position(mm)`) %>%
  mutate(Sample = "BW")

bbdd <- bbdd %>%
  rename(Position = `Position(mm)`) %>%
  mutate(Sample = "bbdd")

# Merge data
df <- bind_rows(BW, bbdd)

# Specify the sample order
df$Sample <- factor(df$Sample, levels = c("BW", "bbdd"))

shift_value   <- 1.0   # Horizontal shift amount
shift_up_value <- 0.05 # Vertical shift amount

df <- df %>%
  mutate(
    Position_plot = ifelse(Sample == "bbdd",
                           Position + shift_value,
                           Position),
    Absorbance_plot = ifelse(Sample == "bbdd",   # ← New
                             Absorbance + shift_up_value,
                             Absorbance)
  )
# ====================================================

# Plot
p <- ggplot(df, aes(x = Position_plot, y = Absorbance_plot, color = Sample)) +
  geom_point(size = 0.2, alpha = 0.8) +
  # geom_line(linewidth = 0.8, alpha = 0.8) +   # Uncomment to connect points
  scale_color_manual(
    values = c("BW" = "#0072B2", "bbdd" = "#D55E00")
  ) +
  labs(
    x = "Position (mm)",
    y = "Absorbance",
    color = "Sample"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black", face = "bold"),
    legend.title = element_text(face = "bold")
  )

p

# Export image
ggsave("Absorbance_H3_plot_shifted.png", p, width = 6, height = 4, dpi = 300)


#fig.5Q
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)

setwd("2020_raw phenotype data")
data <- read.xlsx("ddA276_for_plot.xlsx", 4)
data <- data[-1]
head(data)

# Set material order
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function 
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ==========Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')") 
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +
        #scale_color_manual(values = custom_colors) +
        
        # ========== Statistical significance annotations ==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(1.5, 1.75, 2.0, 1.5, 1.75, 2.0),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 2.5))

# Display plot
print(b)

# Export the figure
ggsave("mutant_45S_ETS_boxplot_4materials.png", b, width = 6, height = 5, dpi = 300)


#fig.5R

#45S_ITS1 expression level visualization

library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)
data <- read.xlsx("ddA276_for_plot.xlsx", 5)
data <- data[-1]
head(data)

data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')") 
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +
        
        # ========== Statistical significance annotations==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(1.5, 1.75, 2.0, 1.5, 1.75, 2.0),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 2.5))

# Display plot
print(b)

# Export the figure
ggsave("mutant_45S_ITS1_boxplot_4materials.png", b, width = 6, height = 5, dpi = 300)

#fig.5S
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)

setwd("2020_raw phenotype data")
data <- read.xlsx("ddA276_for_plot.xlsx", 6)
data <- data[-1]
head(data)

# Change material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function 
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')")  
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +
        
        # ========== Statistical significance annotations==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(1.5, 1.75, 2.0, 1.5, 1.75, 2.0),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 2.5))

# Display plot
print(b)

# Export the figure
ggsave("mutant_45S_junction_boxplot_4materials.png", b, width = 6, height = 4, dpi = 300)


#fig.6A
df <-read.xlsx("A.xlsx",3)
# Calculate density curve data for shaded area
density_data <- density(df$Protein_Content, na.rm = TRUE)
density_df <- data.frame(x = density_data$x, y = density_data$y)

p <-ggplot(df, aes(x = Protein_Content)) +
  # histogramplot
  geom_histogram(aes(y = ..density..), 
                 bins = 50, fill = "#13599F", color = "white", alpha = 0.7) +
  # Add density curve (With shaded area)
  
  geom_line(data = density_df, aes(x = x, y = y), 
            color = "red", size = 1.2) +
  # Add whiskers - above the x-axis
  geom_rug(aes(y = 0), position = position_jitter(height = 0.001), 
           sides = "b", color = "black", alpha = 1, size = 0.5) +
  # Axes and title
  labs(
    x = "Protein Content (%)",
    y = "Density"
    
  ) +
  # theme
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 14, color = "gray40"),
    axis.title = element_text(face = "bold", size = 20),
    axis.text = element_text(size = 18,colour = "black",face = "bold"),
    panel.grid.major = element_blank(),#panel.grid.major = element_line(color = "grey90")this creates a gray grid
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.5)
  )
p
ggsave("Protein_Content_Distribution1.png", 
       plot = p,
       width = 8,      # Width (inches)
       height = 6,     # Height (inches)
       dpi = 300,      # Resolution (dots per inch)
       bg = "white")   # Background color (white)

#fig.6B


df <-read.xlsx("A.xlsx",3)

# Calculate density curve data for shaded area
density_data <- density(df$Carbon_Content, na.rm = TRUE)
density_df <- data.frame(x = density_data$x, y = density_data$y)

p <-ggplot(df, aes(x = Carbon_Content)) +
  # histogramplot
  geom_histogram(aes(y = ..density..), 
                 bins = 50, fill = "#13599F", color = "white", alpha = 0.7) +
  # Add density curve (With shaded area)
 
  geom_line(data = density_df, aes(x = x, y = y), 
            color = "red", size = 1.2) +
  # Add whiskers - above the x-axis
  geom_rug(aes(y = 0), position = position_jitter(height = 0.001), 
           sides = "b", color = "black", alpha = 1, size = 0.5) +
  # Axes and title
  labs(
    x = "Protein Content (%)",
    y = "Density"

  ) +
  # theme
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 14, color = "gray40"),
    axis.title = element_text(face = "bold", size = 20),
    axis.text = element_text(size = 18,colour = "black",face = "bold"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.5)
  )
p
ggsave("Carbon_Content_Distribution1.png", 
       plot = p,
       width = 8,      # Width (inches)
       height = 6,     # Height (inches)
       dpi = 300,      # Resolution (dots per inch)
       bg = "white")   # Background color (white)


#fig.6C

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the data
data = pd.read_excel(r"A.xlsx",
                     sheet_name='Sheet2', header=None)
reflectivity = data.iloc[1:, 0:]

# Read wavelength data
wavelength_df = pd.read_excel(r"wave_length.xlsx", sheet_name='Sheet1')
spectrum = wavelength_df.iloc[:, 1]

# Create plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Plot curves using gradient colors
cmap = plt.cm.Blues
for i in range(len(reflectivity)):
    color = cmap(i / len(reflectivity))
    ax.plot(spectrum, reflectivity.iloc[i],
            color=color, linewidth=0.5, alpha=0.7)

# Plot the mean curve
mean_ref = reflectivity.mean(axis=0)
ax.plot(spectrum, mean_ref, color='red', linewidth=3,
        label=f'Average (n={len(reflectivity)})')

# Set axes
desired_wavelengths = [400, 500, 600, 700, 800, 900, 1000]
tick_positions = []
tick_labels = []

for wl in desired_wavelengths:
    closest_idx = (spectrum - wl).abs().idxmin()
    closest_wl = spectrum[closest_idx]
    if abs(closest_wl - wl) < 20:
        tick_positions.append(closest_wl)
        tick_labels.append(f'{wl}')

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# Labels and title (Bold)
ax.set_xlabel('Wavelength (nm)', fontsize=24, fontweight='bold')
ax.set_ylabel('Reflectance', fontsize=24, fontweight='bold')


# Set range
ax.set_xlim(spectrum.min(), spectrum.max())
ax.set_ylim(0, 0.7)

# Add a legend

legend_props = {'size': 20, 'weight': 'bold'}
ax.legend(fontsize=20, frameon=False, fancybox=True, shadow=True, prop=legend_props)

# Add color bar showing sample density
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(reflectivity)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label('Sample Density', fontsize=24,fontweight='bold')
cbar.ax.tick_params(labelsize=22)

# Make tick labels bold
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Export image
save_path = r"vis_visualization_PLUS1.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='black')


plt.show()

#fig.6D

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the data
data = pd.read_excel(r"A.xlsx",
                     sheet_name='Sheet3', header=None)
reflectivity = data.iloc[1:, 0:]

# Read wavelength data
wavelength_df = pd.read_excel(r"wave_length.xlsx", sheet_name='Sheet3')
spectrum = wavelength_df.iloc[:, 1]

# Create plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Plot curves using gradient colors
cmap = plt.cm.Blues
for i in range(len(reflectivity)):
    color = cmap(i / len(reflectivity))
    ax.plot(spectrum, reflectivity.iloc[i],
            color=color, linewidth=0.5, alpha=0.7)

# Plot the mean curve
mean_ref = reflectivity.mean(axis=0)
ax.plot(spectrum, mean_ref, color='red', linewidth=3,
        label=f'Average (n={len(reflectivity)})')

# Set axes
desired_wavelengths = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
tick_positions = []
tick_labels = []

for wl in desired_wavelengths:
    closest_idx = (spectrum - wl).abs().idxmin()
    closest_wl = spectrum[closest_idx]
    if abs(closest_wl - wl) < 20:
        tick_positions.append(closest_wl)
        tick_labels.append(f'{wl}')

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# Labels and title (Bold)
ax.set_xlabel('Wavelength (nm)', fontsize=24, fontweight='bold')
ax.set_ylabel('Reflectance', fontsize=24, fontweight='bold')

# Set range
ax.set_xlim(spectrum.min(), spectrum.max())
ax.set_ylim(0, 0.7)

# Add a legend

legend_props = {'size': 20, 'weight': 'bold'}
ax.legend(fontsize=20, frameon=False, fancybox=True, shadow=True, prop=legend_props)

# Add color bar showing sample density
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(reflectivity)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label('Sample Density', fontsize=24,fontweight='bold')
cbar.ax.tick_params(labelsize=22)

# Make tick labels bold
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Export image
save_path = r"NIR_visualization_PLUS1.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='black')

plt.show()

#fig.6E
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read correlation data
correlation_df = pd.read_excel(r"correlation visualization data.xlsx")
correlation_df = correlation_df.iloc[:, :3]

# Read wavelength data
wavelength_df = pd.read_excel(r"wave_length.xlsx")

# Merge data, Ensure matching spectral bands
merged_df = pd.merge(correlation_df, wavelength_df, on='band')

# Sort by wavelength
merged_df = merged_df.sort_values('wave_length')

# Create plot
plt.figure(figsize=(12, 8))

# Plot protein correlation line (Black)
plt.plot(merged_df['band'], merged_df.iloc[:, 1],  # Second column: protein correlation
         color='black', linewidth=4,
         label='Protein_content')

# Plot carbon content correlation line (Red)
plt.plot(merged_df['band'], merged_df.iloc[:, 2],  # Third column: carbon content correlation
         color='red', linewidth=4,
         label='Carbon_content')

# Specify wavelength values to display
desired_wavelengths = [400, 500, 600, 700, 800, 900, 1000]

# Found Indices of bands closest to these wavelengths
tick_indices = []
tick_labels = []

for wl in desired_wavelengths:
    # Found Closest wavelength
    idx = (merged_df['wave_length'] - wl).abs().idxmin()
    # only when Found wavelength is sufficiently close to target wavelength
    if abs(merged_df.loc[idx, 'wave_length'] - wl) < 20:  # Allow20nm error
        tick_indices.append(idx)
        tick_labels.append(f'{wl}')

# Set x-axis ticks and labels
plt.xticks(ticks=[list(merged_df['band']).index(merged_df.loc[i, 'band']) for i in tick_indices],
           labels=tick_labels,
           rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# Add title and labels
plt.xlabel('Wavelength (nm)', fontsize=22, fontweight='bold')
plt.ylabel('Correlation', fontsize=22, fontweight='bold')

# Add a legend 
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=24, weight='bold')
plt.legend(loc='best', frameon=False, prop=font_prop)

# Set y axis limits
plt.ylim(-0.2, 0.3)

# automatically Adjust the layout to prevent label clipping
plt.tight_layout()

# Export image to the specified path
save_path = r"vis_correlationvisualization1.png"

# Export image (high Resolution)
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')


# Display plot
plt.show()

#fig.6F

#Visualize correlations between hyperspectral bands and protein/carbon content
#Near infrared

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read correlation data
correlation_df = pd.read_excel(r"correlation visualization data.xlsx")
correlation_df = correlation_df.iloc[:, :3]

# Read wavelength data
wavelength_df = pd.read_excel(r"wave_length.xlsx")

# Merge data, Ensure matching spectral bands
merged_df = pd.merge(correlation_df, wavelength_df, on='band')

# Sort by wavelength
merged_df = merged_df.sort_values('wave_length')

# Create plot
plt.figure(figsize=(12, 8))

# Plot protein correlation line (Black)
plt.plot(merged_df['band'], merged_df.iloc[:, 1],  # Second column: protein correlation
         color='black', linewidth=4,
         label='Protein_content')

# Plot carbon content correlation line (Red)
plt.plot(merged_df['band'], merged_df.iloc[:, 2],  # Third column: carbon content correlation
         color='red', linewidth=4,
         label='Carbon_content')

# Specify wavelength values to display
desired_wavelengths = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]

# Found Indices of bands closest to these wavelengths
tick_indices = []
tick_labels = []

for wl in desired_wavelengths:
    # Found Closest wavelength
    idx = (merged_df['wave_length'] - wl).abs().idxmin()
    # only when Found wavelength is sufficiently close to target wavelength
    if abs(merged_df.loc[idx, 'wave_length'] - wl) < 20:  # Allow20nm error
        tick_indices.append(idx)
        tick_labels.append(f'{wl}')

# Set x-axis ticks and labels
plt.xticks(ticks=[list(merged_df['band']).index(merged_df.loc[i, 'band']) for i in tick_indices],
           labels=tick_labels,
           rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# Add title and labels
plt.xlabel('Wavelength (nm)', fontsize=22, fontweight='bold')
plt.ylabel('Correlation', fontsize=22, fontweight='bold')

# Add a legend 
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=24, weight='bold')
plt.legend(loc='best', frameon=False, prop=font_prop)

# Set y axis limits
plt.ylim(-0.2, 0.3)

# automatically Adjust the layout to prevent label clipping
plt.tight_layout()

# Export image to the specified path
save_path = r"nir_correlationvisualization1.png"

# Export image (high Resolution)
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"plotsheetsavedto: {save_path}")

# Display plot
plt.show()

#fig.6G

library(ggplot2)
library(dplyr)

# ========== protein Data ==========
protein_data <- data.frame(
  Trait = "Protein",
  Variable_Selection = rep(c("Original data", "After CARS", "After UVE", "After SPA"), each = 21),
  Model_Methods = rep(rep(c("PLSR", "PCR", "RR"), each = 7), 4),
  Pretreatment = rep(c("A", "dA", "ddA", "lgA", "MSC", "SG", "SNV"), 12),
  Rp2 = c(
    # Original data
    0.6753, 0.6186, 0.455, 0.6228, 0.5897, 0.6732, 0.5798,   # PLSR
    0.6354, 0.5517, 0.3036, 0.4678, 0.5446, 0.6213, 0.5494,  # PCR
    0.373, 0.336, 0.3017, 0.3179, 0.4215, 0.3925, 0.4278,    # RR
    # After CARS
    0.6668, 0.6806, 0.7488, 0.7983, 0.5138, 0.7586, 0.6785,  # PLSR
    0.5635, 0.54, 0.5158, 0.6372, 0.5088, 0.6271, 0.6112,    # PCR
    0.5514, 0.4994, 0.5509, 0.7814, 0.6977, 0.5308, 0.6499,  # RR
    # After UVE
    0.6931, 0.6732, 0.6587, 0.6538, 0.5996, 0.7029, 0.6093,  # PLSR
    0.6551, 0.6394, 0.5877, 0.6093, 0.51, 0.6226, 0.5563,    # PCR
    0.414, 0.4958, 0.5347, 0.395, 0.4355, 0.3815, 0.439,     # RR
    # After SPA
    0.6445, 0.5634, 0.3725, 0.5398, 0.5665, 0.659, 0.5039,   # PLSR
    0.6466, 0.5104, 0.3338, 0.4554, 0.5963, 0.6028, 0.6304,  # PCR
    0.3701, 0.349, 0.3665, 0.2217, 0.4048, 0.3944, 0.3945    # RR
  )
)

# ========== carbon content Data ==========
carbon_data <- data.frame(
  Trait = "Carbon",
  Variable_Selection = rep(c("Original data", "After CARS", "After UVE", "After SPA"), each = 21),
  Model_Methods = rep(rep(c("PLSR", "PCR", "RR"), each = 7), 4),
  Pretreatment = rep(c("A", "dA", "ddA", "lgA", "MSC", "SG", "SNV"), 12),
  Rp2 = c(
    # Original data
    0.6602, 0.6694, 0.5959, 0.6687, 0.5177, 0.5473, 0.5747,  # PLSR
    0.6214, 0.6606, 0.6468, 0.6094, 0.6019, 0.5854, 0.5806,  # PCR
    0.6229, 0.5944, 0.5235, 0.4757, 0.6139, 0.5645, 0.619,   # RR
    # After CARS
    0.6122, 0.7219, 0.6228, 0.7102, 0.626, 0.5529, 0.5891,   # PLSR
    0.6354, 0.6904, 0.6832, 0.6623, 0.5356, 0.6154, 0.6338,  # PCR
    0.5822, 0.646, 0.4958, 0.4743, 0.4945, 0.5971, 0.3969,   # RR
    # After UVE
    0.7139, 0.694, 0.638, 0.6612, 0.624, 0.5844, 0.5595,     # PLSR
    0.589, 0.7098, 0.7124, 0.6231, 0.5332, 0.5859, 0.5841,   # PCR
    0.6146, 0.5765, 0.5342, 0.5674, 0.612, 0.5689, 0.4673,   # RR
    # After SPA
    0.5675, 0.5455, 0.5166, 0.6744, 0.4712, 0.5111, 0.483,   # PLSR
    0.6274, 0.6101, 0.5702, 0.587, 0.5843, 0.6039, 0.5783,   # PCR
    0.6161, 0.5777, 0.5099, 0.4742, 0.6158, 0.5823, 0.6168   # RR
  )
)

# Set factors
for(data in list(protein_data, carbon_data)) {
  data$Variable_Selection <- factor(data$Variable_Selection,
                                    levels = c("Original data", "After CARS", 
                                               "After UVE", "After SPA"))
  data$Model_Methods <- factor(data$Model_Methods,
                               levels = c("PLSR", "PCR", "RR"))
  data$Pretreatment <- factor(data$Pretreatment,
                              levels = c("A", "dA", "ddA", "lgA", "MSC", "SG", "SNV"))
}

# Color palette
colors_pretreatment <- c(
  "A" = "#d51f26", "d A" = "#89288f", "dd A" = "#8a9fd1", "lg A" = "#c06cab",
  "MSC" = "#f47d2b", "SG" = "#fee500", "SNV" = "#d8a767"
)

# theme
theme_publication <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 10),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10),
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey90", color = "black"),
    strip.text = element_text(face = "bold", size = 11),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.grid.major.y = element_line(color = "grey90", size = 0.3)
  )

# ========== protein plot ==========
p_protein <- ggplot(protein_data, aes(x = Variable_Selection, y = Rp2, fill = Pretreatment)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9),
           color = "black", size = 0.3, width = 0.85) +
  scale_fill_manual(values = colors_pretreatment,
                    name = "Pretreatment") +
  facet_wrap(~Model_Methods, ncol = 3) +
  labs(
    title = expression(paste("Protein Content: Model Performance (", R[p]^2, ")")),
    x = "Variable Selection Method",
    y = expression(paste("Prediction ", R^2))
  ) +
  scale_y_continuous(limits = c(0, 0.85), breaks = seq(0, 0.8, 0.2)) +
  geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 0.5) +
  theme_publication +
  guides(fill = guide_legend(nrow = 1))

# ========== carbon content plot ==========
p_carbon <- ggplot(carbon_data, aes(x = Variable_Selection, y = Rp2, fill = Pretreatment)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9),
           color = "black", size = 0.3, width = 0.85) +
  scale_fill_manual(values = colors_pretreatment,
                    name = "Pretreatment") +
  facet_wrap(~Model_Methods, ncol = 3) +
  labs(
    title = expression(paste("Carbon Content: Model Performance (", R[p]^2, ")")),
    x = "Variable Selection Method",
    y = expression(paste("Prediction ", R^2))
  ) +
  scale_y_continuous(limits = c(0, 0.85), breaks = seq(0, 0.8, 0.2)) +
  geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 0.5) +
  theme_publication +
  guides(fill = guide_legend(nrow = 1))


print(p_protein)
print(p_carbon)

# Merge data
all_data <- rbind(protein_data, carbon_data)

# Set factors
all_data$Trait <- factor(all_data$Trait, levels = c("Protein", "Carbon"))
all_data$Variable_Selection <- factor(all_data$Variable_Selection,
                                      levels = c("Original data", "After CARS", 
                                                 "After UVE", "After SPA"))
all_data$Model_Methods <- factor(all_data$Model_Methods,
                                 levels = c("PLSR", "PCR", "RR"))
all_data$Pretreatment <- factor(all_data$Pretreatment,
                                levels = c("A", "dA", "ddA", "lgA", "MSC", "SG", "SNV"))

# Combined plot (two traits side by side)
p_combined <- ggplot(all_data, aes(x = Variable_Selection, y = Rp2, fill = Pretreatment)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9),
           color = "black", size = 0.2, width = 0.85) +
  scale_fill_manual(values = colors_pretreatment,
                    name = "Pretreatment") +
  facet_grid(Trait ~ Model_Methods, scales = "free_y") +
  labs(
    title = expression(paste("Comparison of Model Performance (", R[p]^2, ") for Different Traits")),
    x = "Variable Selection Method",
    y = expression(paste("Prediction ", R^2))
  ) +
  scale_y_continuous(breaks = seq(0, 0.8, 0.2)) +
  geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 0.4) +
  theme_publication +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
  guides(fill = guide_legend(nrow = 1))

ggsave("Combined_Traits_Model_Comparison.pdf", p_combined, width = 18, height = 10, dpi = 300)
ggsave("Combined_Model_Comparison.png", p_combined, width = 18, height = 10, dpi = 300)

print(p_combined)

# modified of theme - enlarge all fonts、Bold、Black
theme_publication_bold <- theme_classic(base_size = 16) +  
  theme(
    # title - Bold、enlarge、Black
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold", 
      size = 20,      
      color = "black"
    ),
    
    # axis title - Bold、enlarge、Black
    axis.title = element_text(
      face = "bold", 
      size = 18,      
      color = "black"
    ),
    axis.title.x = element_text(
      face = "bold", 
      size = 18, 
      color = "black"
    ),
    axis.title.y = element_text(
      face = "bold", 
      size = 18, 
      color = "black"
    ),
    
    # axis tick labels - Bold、enlarge、Black
    axis.text = element_text(
      color = "black", 
      size = 14,      
      face = "bold"   # Add Bold
    ),
    axis.text.x = element_text(
      angle = 45, 
      hjust = 1, 
      vjust = 1,
      size = 14,      
      face = "bold",
      color = "black"
    ),
    axis.text.y = element_text(
      size = 14,
      face = "bold",
      color = "black"
    ),
    
    # legend - Bold、enlarge、Black
    legend.title = element_text(
      face = "bold", 
      size = 16,      # 
      color = "black"
    ),
    legend.text = element_text(
      size = 14,      # from10changeas14
      face = "bold",  # Add Bold
      color = "black"
    ),
    legend.position = "bottom",
    
    # facet labels - Bold、enlarge、Black
    strip.background = element_rect(
      fill = "grey90", 
      color = "black",
      size = 1
    ),
    strip.text = element_text(
      face = "bold", 
      size = 16,      # 
      color = "black"
    ),
    strip.text.x = element_text(
      face = "bold", 
      size = 16, 
      color = "black"
    ),
    strip.text.y = element_text(
      face = "bold", 
      size = 16, 
      color = "black"
    ),
    
    # panel border
    panel.border = element_rect(
      color = "black", 
      fill = NA, 
      size = 1.2      # border Bold
    ),
    panel.grid.major.y = element_line(
      color = "grey90", 
      size = 0.3
    )
  )

# Combined plot (two traits side by side)
p_combined <- ggplot(all_data, aes(x = Variable_Selection, y = Rp2, fill = Pretreatment)) +
  geom_bar(stat = "identity", 
           position = position_dodge(width = 0.9),
           color = "black", 
           size = 0.4,      # bars border
           width = 0.85) +
  scale_fill_manual(
    values = colors_pretreatment,
    name = "Pretreatment Method"
  ) +
  facet_grid(Trait ~ Model_Methods, scales = "free_y") +
  labs(
    title = expression(bold(paste("Comparison of Model Performance (", R[p]^2, ") for Different Traits"))),
    x = expression(bold("Variable Selection Method")),
    y = expression(bold(paste("Prediction ", R^2)))
  ) +
  scale_y_continuous(breaks = seq(0, 0.8, 0.2)) +
  geom_hline(yintercept = 0.7, 
             linetype = "dashed", 
             color = "red", 
             size = 0.8) +    # reference line
  theme_publication_bold +
  guides(fill = guide_legend(
    nrow = 1,
    keywidth = unit(1.5, "cm"),   # legend enlarge boxes
    keyheight = unit(0.8, "cm")
  ))

# Display plot
print(p_combined)

# Export
ggsave("Combined_Traits_Model_Comparison_Bold.pdf", 
       p_combined, 
       width = 10,      
       height = 6,    
       dpi = 300)

ggsave("Combined_Traits_Model_Comparison_Bold.png", 
       p_combined, 
       width = 12, 
       height = 6, 
       dpi = 300)




#fig.6H
library(ggplot2)
library(ggExtra)
library(patchwork)
library(dplyr)

# ========== 1. Read or Prepare the input data ==========
dat <-read.xlsx("model_results.xlsx",1)

# Ensure Group is a factor
dat$Group <- factor(dat$Group)

# check Data
head(dat)
str(dat)

# ========== 2. Calculate correlation coefficients by group & text ==========
# Compute correlations for the training and test sets
cors_train <- cor.test(dat$Y_true[dat$Group == "Training_Set"], 
                       dat$Y_pred[dat$Group == "Training_Set"])

cors_test <- cor.test(dat$Y_true[dat$Group == "Test_Set"], 
                      dat$Y_pred[dat$Group == "Test_Set"])

# Generate correlation coefficient labels
generate_cor_text <- function(cor_result, group_name) {
  estimate <- round(cor_result$estimate, 3)
  p_value <- cor_result$p.value
  
  if (p_value < 0.001) {
    p_text <- "p < 0.001"
  } else if (p_value < 0.01) {
    p_text <- paste0("p = ", round(p_value, 3))
  } else if (p_value < 0.05) {
    p_text <- paste0("p = ", round(p_value, 3))
  } else {
    p_text <- paste0("p = ", round(p_value, 3))
  }
  
  return(paste0(group_name, ": R = ", estimate, ", ", p_text))
}

# generate text
txt_train <- generate_cor_text(cors_train, "Training Set")
txt_test <- generate_cor_text(cors_test, "Test Set")

# ========== 3. Create base plot ==========
# Define colors
train_color <- "#1f77b4" # dark Blue
test_color <- "#d62728" # cyan

base <- ggplot(dat, aes(x = Y_true, y = Y_pred, 
                        colour = Group, fill = Group)) +
  # Add Scatter points
  geom_point(alpha = 0.7, size = 2) +
  # Add regression line and confidence interval
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2, size = 0.8) +
  # Add y=x reference line (Ideal prediction line)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", size = 0.6, alpha = 0.7) +
  # Set colors
  scale_colour_manual(values = c("Training_Set" = train_color, 
                                 "Test_Set" = test_color)) +
  scale_fill_manual(values = c("Training_Set" = train_color, 
                               "Test_Set" = test_color)) +
  # Axis labels
  labs(x = "Y_true", 
       y = "Y_predicted"
  ) +
  # Add correlation coefficient text
  annotate("text", x = min(dat$Y_true, na.rm = TRUE), 
           y = max(dat$Y_pred, na.rm = TRUE) * 1.04,
           label = txt_train, hjust = 0, 
           colour = train_color, size = 6, fontface = "bold") +
  annotate("text", x = min(dat$Y_true, na.rm = TRUE), 
           y = max(dat$Y_pred, na.rm = TRUE) * 1.00,
           label = txt_test, hjust = 0, 
           colour = test_color, size = 6, fontface = "bold") +
  
  # Theme settings
  theme_classic() +
  theme(
    panel.border = element_rect(fill = "NA", color = "black", size = 0.8),
    axis.line = element_line(size = 0.6, colour = "black"),
    legend.position = "none",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11,face = "bold"),
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 16,face = "bold")
  )

# Display plot
print(base)

# ========== Add marginal distribution plots ==========

base_with_margins <- ggMarginal(
  base,
  type = "density",  # or "histogram"
  groupColour = TRUE,
  groupFill = TRUE,
  alpha = 0.5
)

print(base_with_margins)

# ========== Export the figure ==========

# Export plot with marginal distributions
ggsave("protein_prediction1.png", base_with_margins, 
       width = 9, height = 6, dpi = 300)


#fig.6I
library(ggplot2)
library(ggExtra)
library(patchwork)
library(dplyr)

# ========== 1. Read or Prepare the input data ==========
dat <-read.xlsx("model_results.xlsx",2)

# Ensure Groupis a factor
dat$Group <- factor(dat$Group)

# check Data
head(dat)
str(dat)

# ========== 2. Calculate correlation coefficients by group & text ==========
# Compute correlations for the training and test sets
cors_train <- cor.test(dat$Y_true[dat$Group == "Training_Set"], 
                       dat$Y_pred[dat$Group == "Training_Set"])

cors_test <- cor.test(dat$Y_true[dat$Group == "Test_Set"], 
                      dat$Y_pred[dat$Group == "Test_Set"])

# Generate correlation coefficient labels
generate_cor_text <- function(cor_result, group_name) {
  estimate <- round(cor_result$estimate, 3)
  p_value <- cor_result$p.value
  
  if (p_value < 0.001) {
    p_text <- "p < 0.001"
  } else if (p_value < 0.01) {
    p_text <- paste0("p = ", round(p_value, 3))
  } else if (p_value < 0.05) {
    p_text <- paste0("p = ", round(p_value, 3))
  } else {
    p_text <- paste0("p = ", round(p_value, 3))
  }
  
  return(paste0(group_name, ": R = ", estimate, ", ", p_text))
}

# generate text
txt_train <- generate_cor_text(cors_train, "Training Set")
txt_test <- generate_cor_text(cors_test, "Test Set")

# ========== 3. Create base plot ==========
# Define colors
train_color <- "#333463" # dark Blue
test_color <- "#079C9C" # cyan

base <- ggplot(dat, aes(x = Y_true, y = Y_pred, 
                        colour = Group, fill = Group)) +
  # Add Scatter points
  geom_point(alpha = 0.7, size = 2) +
  # Add regression line and confidence interval
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2, size = 0.8) +
  # Add y=x reference line (Ideal prediction line)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", size = 0.6, alpha = 0.7) +
  # Set colors
  scale_colour_manual(values = c("Training_Set" = train_color, 
                                 "Test_Set" = test_color)) +
  scale_fill_manual(values = c("Training_Set" = train_color, 
                               "Test_Set" = test_color)) +
  # Axis labels
  labs(x = "Y_true", 
       y = "Y_predicted"
       ) +
  # Add correlation coefficient text
  annotate("text", x = min(dat$Y_true, na.rm = TRUE), 
           y = max(dat$Y_pred, na.rm = TRUE) * 1.04,
           label = txt_train, hjust = 0, 
           colour = train_color, size = 6, fontface = "bold") +
  annotate("text", x = min(dat$Y_true, na.rm = TRUE), 
           y = max(dat$Y_pred, na.rm = TRUE) * 1.03,
           label = txt_test, hjust = 0, 
           colour = test_color, size = 6, fontface = "bold") +
  
  # Theme settings
  theme_classic() +
  theme(
    panel.border = element_rect(fill = "NA", color = "black", size = 0.8),
    axis.line = element_line(size = 0.6, colour = "black"),
    legend.position = "none",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11,face = "bold"),
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),
    axis.text = element_text(size = 16,face = "bold")
  )

# Display plot
print(base)

# ========== Add marginal distribution plots ==========

base_with_margins <- ggMarginal(
  base,
  type = "density",  # or "histogram"
  groupColour = TRUE,
  groupFill = TRUE,
  alpha = 0.5
)

print(base_with_margins)

# ========== Export the figure ==========

# Export plot with marginal distributions
ggsave("carbon_prediction1.png", base_with_margins, 
       width = 9, height = 6, dpi = 300)

#fig.6J
# Load required packages
library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggsignif)
library(ggbeeswarm)
library(patchwork)

# Specify global font sizes and theme
theme_set(theme_bw(base_size = 16) + 
            theme(
              plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
              axis.title.x = element_text(size = 26, face = "bold"),
              axis.title.y = element_text(size = 26, face = "bold"),
              axis.text.x = element_text(size = 22, color = "black", face = "bold"),
              axis.text.y = element_text(size = 22, color = "black", face = "bold"),
              legend.position = "none",
              plot.margin = margin(20, 20, 20, 20),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_rect(color = "black", fill = NA, size = 1),
              axis.ticks = element_line(color = "black", size = 0.8)
            ))

# Read the Excel file
df_file <- "seedHSI.xlsx"
df <- read.xlsx(df_file, sheet = 'SUM')
df <- df[, 1:2]

# rename column for processing
#colnames(df) <- c("C_Protein", "H_Protein")

# Reshape data into long format
data_long <- df %>%
  pivot_longer(
    cols = everything(),
    names_to = "Group",
    values_to = "Protein"
  ) %>%
  mutate(
    Group = gsub("_Protein", "", Group)
  )

# extract C group and H group of Data used for ttest
C_Protein <- na.omit(df$C_Protein)
H_Protein <- na.omit(df$H_Protein)

# perform ttest
t_test_result <- t.test(C_Protein, H_Protein)
t_stat <- t_test_result$statistic
p_value <- t_test_result$p.value

# Calculate group statistics
stats_summary <- data_long %>%
  group_by(Group) %>%
  summarise(
    mean = mean(Protein, na.rm = TRUE),
    sd = sd(Protein, na.rm = TRUE),
    n = n()
  )

# According to pvalue to determine significance labels
sig_symbol <- ifelse(p_value < 0.001, "bold('***')",
                     ifelse(p_value < 0.01, "bold('**')",
                            ifelse(p_value < 0.05, "bold('*')", "bold('ns')")))

# Create plot
p_ggsignif <- ggplot(data_long, aes(x = Group, y = Protein, color = Group)) +
  geom_beeswarm(
    size = 3,
    cex = 1.5,
    alpha = 0.8
  ) +
  scale_color_manual(
    values = c("C" = "#1f77b4", "H" = "#d62728")
  ) +
  labs(
    x = "Group",
    y = "Protein_Content(%)"
  ) +
  # Use ggsignif Add significance markers
  geom_signif(
    comparisons = list(c("C", "H")),
    annotations = sig_symbol,
    y_position = max(data_long$Protein, na.rm = TRUE) * 1.1,
    tip_length = 0.01,
    textsize = 12,
    vjust = 0.2,
    color = "black",
    parse = TRUE
  ) +
  # Annotate the plot with p-values
  annotate("text", 
           x = 1.5, 
           y = max(data_long$Protein, na.rm = TRUE) * 1.3,
           label = sprintf("p = %.4f", p_value),
           size = 9,
           hjust = 0.5,
           fontface = "bold")

print(p_ggsignif)

# Export as PNG format
ggsave(
  filename = file.path("protein1.png"),
  plot = p_ggsignif,
  width = 12,
  height = 8,
  dpi = 300,
  device = "png"
)


#fig.6K

#498 copies Predict carbon content visualization

# Load required packages
library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggsignif)
library(ggbeeswarm)
library(patchwork)

# Specify global font sizes and theme
theme_set(theme_bw(base_size = 16) + 
            theme(
              plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
              axis.title.x = element_text(size = 26, face = "bold"),
              axis.title.y = element_text(size = 26, face = "bold"),
              axis.text.x = element_text(size = 22, color = "black", face = "bold"),
              axis.text.y = element_text(size = 22, color = "black", face = "bold"),
              legend.position = "none",
              plot.margin = margin(20, 20, 20, 20),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_rect(color = "black", fill = NA, size = 1),
              axis.ticks = element_line(color = "black", size = 0.8)
            ))

# Read the Excel file
df_file <- "/2020 year grain quality Predict.xlsx"
df <- read.xlsx(df_file, sheet = 'SUM')
df <- df[, 3:4]


# Reshape data into long format
data_long <- df %>%
  pivot_longer(
    cols = everything(),
    names_to = "Group",
    values_to = "Carbon"
  ) %>%
  mutate(
    Group = gsub("_Carbon", "", Group)
  )

# extract C group and H group of Data used for ttest
C_Carbon <- na.omit(df$C_Carbon)
H_Carbon <- na.omit(df$H_Carbon)

# perform ttest
t_test_result <- t.test(C_Carbon, H_Carbon)
t_stat <- t_test_result$statistic
p_value <- t_test_result$p.value

# Calculate group statistics
stats_summary <- data_long %>%
  group_by(Group) %>%
  summarise(
    mean = mean(Carbon, na.rm = TRUE),
    sd = sd(Carbon, na.rm = TRUE),
    n = n()
  )

# According to pvalue to determine significance labels
sig_symbol <- ifelse(p_value < 0.001, "bold('***')",
                     ifelse(p_value < 0.01, "bold('**')",
                            ifelse(p_value < 0.05, "bold('*')", "bold('ns')")))

# Create plot
p_ggsignif <- ggplot(data_long, aes(x = Group, y = Carbon, color = Group)) +
  geom_beeswarm(
    size = 3,
    cex = 1.5,
    alpha = 0.8
  ) +
  scale_color_manual(
    values = c("C" = "#1f77b4", "H" = "#d62728")
  ) +
  labs(
    x = "Group",
    y = "Carbon_Content(%)"
  ) +
  # Use ggsignif Add significance markers
  geom_signif(
    comparisons = list(c("C", "H")),
    annotations = sig_symbol,
    y_position = max(data_long$Carbon, na.rm = TRUE) * 1.1,
    tip_length = 0.01,
    textsize = 12,
    vjust = 0.2,
    color = "black",
    parse = TRUE
  ) +
  # Annotate the plot with p-values
  annotate("text", 
           x = 1.5, 
           y = max(data_long$Carbon, na.rm = TRUE) * 1.3,
           label = sprintf("p = %.4f", p_value),
           size = 9,
           hjust = 0.5,
           fontface = "bold")

print(p_ggsignif)

# Export as PNG format (high quality)
ggsave(
  filename = file.path("Carbon1.png"),
  plot = p_ggsignif,
  width = 12,
  height = 8,
  dpi = 300,
  device = "png"
)


#Fig.S2
# QTL length distribution histogram plot
library(ggplot2)
library(dplyr)
library(openxlsx)
# ========== Import the data ==========

qtl_file <- "QTL.xlsx"

# Import the data
df <- read.xlsx(qtl_file, sheet = 1)

cat("Datadimension:", dim(df), "\n")
cat("Column names:", colnames(df), "\n")

# ========== Process data ==========

# Calculate QTL length (Mb)
df <- df %>%
  mutate(Length_Mb = (RIGHT - LEFT) / 1e6)

# Ensure Groupis a factor
df$Group <- factor(df$Group, levels = c("C", "H"))

# inspect Data summary
cat("\nData summary:\n")
summary_stats <- df %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Min = min(Length_Mb),
    Max = max(Length_Mb),
    Mean = mean(Length_Mb),
    Median = median(Length_Mb)
  )
print(summary_stats)

# ========== Color palette ==========

color_c <- "#1f77b4" # Blue
color_h <- '#d62728' # Red

custom_colors <- c("C" = color_c, "H" = color_h)

# ========== draw basic histogram plot ==========

p <- ggplot(df, aes(x = Length_Mb, fill = Group)) +
  
  # histogramplot
  geom_histogram(
    position = "stack",  # overlay
    alpha = 1,           # transparency
    binwidth = 0.5,        # bars Width (adjustable)
    color = "white",       # bars border color
    linewidth = 0.8        # border thickness
  ) +
  
  # color
  scale_fill_manual(
    values = custom_colors,
    labels = c("C" = "Control (C)", "H" = "Drought (H)")
  )

# display basic plot
print(p)

# ==========  Add Axis labels ==========

p <- p +
  labs(
    x = "QTL Length (Mb)",
    y = "Frequency",
    title = "Distribution of QTL Lengths",
    fill = ""
  )

print(p)

# ========== Refine the plot theme ==========

p <- p +
  theme_bw() +
  theme(
    # title
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold", color = "black"),
    axis.title.x = element_text(size = 11, face = "bold", color = "black"),
    # , face = "bold", color = "black"),
    axis.title.y = element_text(size = 11, face = "bold", color = "black"),
    
    # axis ticks
    axis.text.x = element_text(size = 9, color = "black",face = "bold"),
    axis.text.y = element_text(size = 9, color = "black",face = "bold"),
    
    # legend
    legend.position = c(0.80, 0.75),  # position: upper right
    legend.title = element_blank(),
    legend.text = element_text(size = 9, color = "black",face = "bold"),
    legend.background = element_rect(fill = "white", color = NA, linewidth = 0),
    legend.key.size = unit(0.8, "cm"),
    
    # grid
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    
    # border
    panel.border = element_rect(color = "black", linewidth = 1)
  )

print(p)

# ========== adjust Y axis limits ==========

p <- p +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

print(p)

# ========== Export image ==========

# PNG format
ggsave("QTL_Length_Distribution.png", p, width = 4.5, height = 3, dpi = 300)
cat("\n✓ PNGsaved\n")

# PDF format
ggsave("QTL_Length_Distribution.pdf", p, width = 8, height = 5)
cat("✓ PDFsaved\n")


#Fig.S3

# QTL-trait association network plot visualization

library(igraph)
library(ggraph)
library(tidygraph)
library(dplyr)
library(openxlsx)
library(ggplot2)

# ========== Specify the working directory ==========


# Import the data
qtl_summary <- read.xlsx("QTL_Trait_Association_Statistics.xlsx", sheet = 4)

# filter multiple traits QTL (association 2 items or more trait)
multi_trait_qtls <- qtl_summary %>%
  filter(N_Traits >= 2) %>%
  select(
    Merged_QTL_ID,
    QTL_Type,
    Treatment,
    Chromosome,
    N_Traits,
    Trait_List
  )

cat("multiple traits QTL count:", nrow(multi_trait_qtls), "\n\n")

# ========== abbreviate QTL labels, retain Common as the label ==========

# group by type, as Each QTL assign short IDs
multi_trait_qtls <- multi_trait_qtls %>%
  group_by(QTL_Type) %>%
  mutate(
    Short_Name = case_when(
      QTL_Type == "Common" ~ paste0("Common-", row_number()),
      QTL_Type == "C_Specific" ~ paste0("C-", row_number()),
      QTL_Type == "H_Specific" ~ paste0("H-", row_number()),
      TRUE ~ as.character(row_number())
    )
  ) %>%
  ungroup()

# create name mapping table, for later lookup
qtl_name_mapping <- multi_trait_qtls %>%
  select(
    Original_Name = Merged_QTL_ID,
    Short_Name,
    QTL_Type,
    Chromosome,
    N_Traits,
    Trait_List
  )

cat("\n QTL name mapping example:\n")
print(head(qtl_name_mapping, 10))


# ========== construct network Data, Use short names ==========

edges_list <- list()

for (i in 1:nrow(multi_trait_qtls)) {
  
  qtl_short_name <- multi_trait_qtls$Short_Name[i]
  qtl_type <- multi_trait_qtls$QTL_Type[i]
  
  # split traits column table
  traits <- strsplit(as.character(multi_trait_qtls$Trait_List[i]), ";")[[1]]
  
  # remove possible in of leading and trailing spaces
  traits <- trimws(traits)
  
  # as Each trait create an edge
  for (trait in traits) {
    edges_list[[length(edges_list) + 1]] <- data.frame(
      from = qtl_short_name,
      to = trait,
      QTL_Type = qtl_type,
      stringsAsFactors = FALSE
    )
  }
}

edges <- do.call(rbind, edges_list)

cat("\nedge（connect）count:", nrow(edges), "\n")

# ========== create nodes Data box ==========

qtl_nodes <- multi_trait_qtls %>%
  select(
    name = Short_Name,
    QTL_Type,
    N_Traits
  ) %>%
  mutate(
    node_type = "QTL",
    size = N_Traits
  )

trait_nodes <- data.frame(
  name = unique(edges$to),
  QTL_Type = NA,
  N_Traits = NA,
  node_type = "Trait",
  size = 3,
  stringsAsFactors = FALSE
)

nodes <- rbind(
  qtl_nodes,
  trait_nodes
)

cat("number of nodes:", nrow(nodes), "\n")
cat("  - QTL node:", sum(nodes$node_type == "QTL"), "\n")
cat("  - trait nodes:", sum(nodes$node_type == "Trait"), "\n\n")

# ========== create network plotobject ==========

graph <- graph_from_data_frame(
  d = edges,
  vertices = nodes,
  directed = FALSE
)

# Calculate node degree
node_degree <- degree(graph)

# as node Add degree and label information

nodes_with_info <- nodes %>%
  mutate(
    degree = node_degree[match(name, names(node_degree))],
    degree = ifelse(is.na(degree), 0, degree),
    
    # initial label display condition
    show_label = case_when(
      node_type == "QTL" & N_Traits >= 4 ~ TRUE,
      node_type == "Trait" & degree >= 3 ~ TRUE,
      TRUE ~ FALSE
    ),
    label_text = ifelse(show_label, name, "")
  )

# ========== create unified of node categories column ==========

nodes_with_info <- nodes_with_info %>%
  mutate(
    node_category = ifelse(
      node_type == "QTL",
      as.character(QTL_Type),
      "Trait"
    )
  )

# ========== legend display Common ==========

nodes_with_info$node_category <- factor(
  nodes_with_info$node_category,
  levels = c("Common", "C_Specific", "H_Specific", "Trait"),
  labels = c("Common", "C_Specific", "H_Specific", "Trait")
)

# ========== strict of label display conditions ==========

nodes_with_info <- nodes_with_info %>%
  mutate(
    show_label = case_when(
      # QTLnode: only display associated ≥ 5items trait of
      node_type == "QTL" & N_Traits >= 5 ~ TRUE,
      
      # trait nodes: display only connections ≥ 4 QTLs
      node_type == "Trait" & degree >= 4 ~ TRUE,
      
      TRUE ~ FALSE
    ),
    label_text = ifelse(show_label, name, "")
  )

# ========== update plot object ==========

graph_final <- graph_from_data_frame(
  d = edges,
  vertices = nodes_with_info,
  directed = FALSE
)

tbl_graph_final <- as_tbl_graph(graph_final)

# ========== Define the color palette ==========

node_colors <- c(
  "Common" = "#1E8099", # Common QTL
  "C_Specific" = "#25C1BE", # C-Specific QTL
  "H_Specific" = "#A8CB39", # H-Specific QTL
  "Trait" = "#B13774" # Trait
)

# ========== draw network plot ==========

p1 <- ggraph(tbl_graph_final, layout = "fr") +
  
  # edge
  geom_edge_link(
    alpha = 0.3,
    width = 0.6,
    color = "grey60"
  ) +
  
  # node
  geom_node_point(
    aes(
      size = ifelse(node_type == "QTL", size * 3, 6),
      color = node_category,
      shape = node_type
    ),
    stroke = 0
  ) +
  
  # color legend
  scale_color_manual(
    values = node_colors,
    name = "",
    labels = c(
      "Common" = "Common QTL",
      "C_Specific" = "C-Specific QTL",
      "H_Specific" = "H-Specific QTL",
      "Trait" = "Trait"
    ),
    breaks = c("Common", "C_Specific", "H_Specific", "Trait"),
    drop = FALSE
  ) +
  
  # shape mapping
  scale_shape_manual(
    values = c(
      "QTL" = 19,
      "Trait" = 15
    ),
    guide = "none"
  ) +
  
  # node label
  geom_node_text(
    aes(label = label_text),
    size = 3.5,
    repel = TRUE,
    max.overlaps = 30,
    fontface = "bold",
    color = "black",
    bg.color = "white",
    bg.r = 0.15,
    segment.color = "grey50",
    segment.size = 0.3
  ) +
  
  # node size
  scale_size_continuous(
    range = c(3, 15),
    guide = "none"
  ) +
  
  # theme
  theme_graph() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 14,
      face = "bold"
    ),
    plot.subtitle = element_text(
      hjust = 0.5,
      size = 11
    ),
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(
      size = 10,
      face = "bold"
    ),
    legend.key = element_rect(
      fill = "white",
      color = NA
    ),
    legend.key.size = unit(1, "cm"),
    legend.spacing.y = unit(0.3, "cm")
  ) +
  
  # legend style
  guides(
    color = guide_legend(
      override.aes = list(
        size = 6,
        shape = c(19, 19, 19, 15)
      )
    )
  ) +
  
  labs(
    title = "QTL-Trait Association Network",
    subtitle = paste0(
      "Showing ",
      sum(nodes_with_info$show_label & nodes_with_info$node_type == "QTL"),
      " major QTLs and ",
      sum(nodes_with_info$show_label & nodes_with_info$node_type == "Trait"),
      " key traits"
    )
  )

print(p1)

# ========== Export image ==========

ggsave(
  "QTL_Trait_Network_Final.png",
  p1,
  width = 16,
  height = 12,
  dpi = 600
)

ggsave(
  "QTL_Trait_Network_Final.pdf",
  p1,
  width = 16,
  height = 12
)


#Fig.S4A

#7 items gene of expression level Plot

# Load required packages
library(ggplot2)
library(tidyr)
library(dplyr)
library(openxlsx)

# ========== Prepare the input data ==========

# Import the data
gene_data <- read.xlsx("gene_expression.xlsx", 1)

# Convert to long format
data_long <- gene_data %>%
  pivot_longer(
    cols = -Category,
    names_to = "Sample_Type",
    values_to = "Expression"
  )

# ========== modify Sample_Type name, Add underscore ==========
data_long <- data_long %>%
  mutate(
    Sample_Type = case_when(
      Sample_Type == "embryo14dpa" ~ "embryo_14dpa",
      Sample_Type == "endosperm14dpa" ~ "endosperm_14dpa",
      Sample_Type == "embryo25dpa" ~ "embryo_25dpa",
      Sample_Type == "endosperm25dpa" ~ "endosperm_25dpa",
      TRUE ~ Sample_Type  # keep other value unchanged
    )
  )

# Set factor order
data_long$Sample_Type <- factor(data_long$Sample_Type,
                                levels = c("embryo_14dpa", "endosperm_14dpa", 
                                           "embryo_25dpa", "endosperm_25dpa"))

data_long$Category <- factor(data_long$Category, 
                             levels = unique(gene_data$Category))
data_long$Expression <- as.numeric(data_long$Expression)

# Compute the mean and standard error
stat_data <- data_long %>%
  group_by(Sample_Type, Category) %>%
  summarise(
    n = n(),
    mean = mean(Expression),
    sd = sd(Expression),
    se = sd(Expression) / sqrt(n),
    .groups = "drop"
  )

cat("Data preparation completed\n")
print(head(stat_data))

# ========== Define colors ==========

gene_colors <- c(
  "TraesCS2B03G0416500" = "#d51f26",
  "TraesCS2B03G0415300" = "#272e6a",
  "TraesCS2B03G0415600" = "#208a42",
  "TraesCS2B03G0415800" = "#89288f",
  "TraesCS2B03G0414600" = "#f47d2b",
  "TraesCS2B03G0415900" = "#fee500",
  "TraesCS2B03G0415100" = "#8a9fd1"
)

# ========== Custom plotting theme ==========

custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black", 
                               angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ========== Plotting ==========

p <- ggplot(stat_data, aes(x = Sample_Type, y = mean, fill = Category)) +
  
  # 1. Draw bar chart
  geom_bar(
    stat = "identity",
    width = 0.6,
    position = position_dodge(width = 0.7),
    alpha = 0.8
  ) +
  
  # 2. Add error bars (standard error)
  geom_errorbar(
    aes(ymin = mean - se, ymax = mean + se, color = Category),
    width = 0.2,
    position = position_dodge(width = 0.7),
    size = 1,
    show.legend = FALSE
  ) +
  
  # 3. Add raw data points
  geom_point(
    data = data_long,
    aes(x = Sample_Type, y = Expression, color = Category),
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 2,
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  # 4. Color settings
  scale_fill_manual(values = gene_colors, name = "Gene") +
  scale_color_manual(values = gene_colors, name = "Gene") +
  
  # 5. Axis labels
  labs(
    x = "Sample Type",
    y = "Expression Level"
  )

# Apply the plotting theme
b <- p + custom_theme_advanced

# Display plot
print(b)

# ========== Export the figure ==========

ggsave(
  filename = "Gene_Expression_Barplot.png",
  plot = b,
  width = 10,
  height = 6,
  dpi = 300
)

ggsave(
  filename = "Gene_Expression_Barplot.pdf",
  plot = b,
  width = 12,
  height = 7
)

ggsave(
  filename = "Gene_Expression_Barplot.tiff",
  plot = b,
  width = 12,
  height = 7,
  dpi = 600,
  compression = "lzw"
)

##Fig.S4B

#after flowering 12 days 7 items gene of expression level
# Load required packages
library(ggplot2)
library(tidyr)
library(dplyr)
library(openxlsx)

# ========== Prepare the input data ==========

# Import the data
gene_data <- read.xlsx("7 items gene of expression level Plot Data.xlsx", 2)


# Convert to long format
data_long <- gene_data %>%
  pivot_longer(
    cols = -Category,
    names_to = "Tissue_Type",
    values_to = "Expression"
  )

# Set factor order
data_long$Tissue_Type <- factor(data_long$Tissue_Type,
                                levels = c("endosperm_12DPA", "inner_pericarp_12DPA", 
                                           "outer_pericarp_12DPA"))

data_long$Category <- factor(data_long$Category, 
                             levels = unique(gene_data$Category))
data_long$Expression <- as.numeric(data_long$Expression)

# Compute the mean and standard error
stat_data <- data_long %>%
  group_by(Tissue_Type, Category) %>%
  summarise(
    n = n(),
    mean = mean(Expression, na.rm = TRUE),
    sd = sd(Expression, na.rm = TRUE),
    se = sd(Expression, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nData preparation completed\n")
cat("statistics Data:\n")
print(stat_data)

# ========== Define colors ==========

gene_colors <- c(
  "TraesCS2B03G0416500" = "#d51f26",
  "TraesCS2B03G0415300" = "#272e6a",
  "TraesCS2B03G0415600" = "#208a42",
  "TraesCS2B03G0415800" = "#89288f",
  "TraesCS2B03G0414600" = "#f47d2b",
  "TraesCS2B03G0415900" = "#fee500",
  "TraesCS2B03G0415100" = "#8a9fd1"
)

# ========== Custom plotting theme ==========

custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 14, face = "bold", color = "black", 
                               angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ========== Plotting ==========

p <- ggplot(stat_data, aes(x = Tissue_Type, y = mean, fill = Category)) +
  
  # 1. Draw bar chart
  geom_bar(
    stat = "identity",
    width = 0.6,
    position = position_dodge(width = 0.7),
    alpha = 0.8
  ) +
  
  # 2. Add error bars (standard error)
  geom_errorbar(
    aes(ymin = mean - se, ymax = mean + se, color = Category),
    width = 0.2,
    position = position_dodge(width = 0.7),
    size = 1,
    show.legend = FALSE
  ) +
  
  # 3. Add raw data points
  geom_point(
    data = data_long,
    aes(x = Tissue_Type, y = Expression, color = Category),
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 2,
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  # 4. Color settings
  scale_fill_manual(values = gene_colors, name = "Gene") +
  scale_color_manual(values = gene_colors, name = "Gene") +
  
  # 5. Axis labels
  labs(
    x = "Tissue Type",
    y = "Expression Level"
  )

# Apply the plotting theme
b <- p + custom_theme_advanced

# Display plot
print(b)

# ========== Export the figure ==========

ggsave(
  filename = "Gene_Expression_12DPA_Tissues.png",
  plot = b,
  width = 10,
  height = 6,
  dpi = 300
)

ggsave(
  filename = "Gene_Expression_12DPA_Tissues.pdf",
  plot = b,
  width = 12,
  height = 7
)

ggsave(
  filename = "Gene_Expression_12DPA_Tissues.tiff",
  plot = b,
  width = 12,
  height = 7,
  dpi = 600,
  compression = "lzw"
)

#Fig.S4C

#7 items gene expression level under different stress 
# Load required packages
library(ggplot2)
library(tidyr)
library(dplyr)
library(openxlsx)

# ========== Prepare the input data ==========

# Import the data
gene_data <- read.xlsx("7itemsgeneofexpression levelPlotData.xlsx", 3)


# ========== first all column (except Category outside)convert as number value type ==========

# Export Category column
category_col <- gene_data$Category

# convert except No.one column outside of all column as number value type
for(i in 2:ncol(gene_data)) {
  gene_data[[i]] <- as.numeric(gene_data[[i]])
}

# restore Category column
gene_data$Category <- category_col

cat("\nconvertafterofDatatype:\n")
print(sapply(gene_data, class))

# Convert to long format
data_long <- gene_data %>%
  pivot_longer(
    cols = -Category,
    names_to = "Treatment",
    values_to = "Expression"
  )

# Set factor order
data_long$Treatment <- factor(data_long$Treatment,
                              levels = c("control", "drought_1h", "drought_6h", 
                                         "heat_1h", "heat_6h", 
                                         "drought&heat_1h", "drought&heat_6h"))

data_long$Category <- factor(data_long$Category, 
                             levels = unique(gene_data$Category))

# Ensure Expression is numeric value type
data_long$Expression <- as.numeric(data_long$Expression)

# Compute the mean and standard error
stat_data <- data_long %>%
  group_by(Treatment, Category) %>%
  summarise(
    n = n(),
    mean = mean(Expression, na.rm = TRUE),
    sd = sd(Expression, na.rm = TRUE),
    se = sd(Expression, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nData preparation completed\n")
cat("statistics Data:\n")
print(head(stat_data, 14))

# ========== Define colors ==========

gene_colors <- c(
  "TraesCS2B03G0416500" = "#d51f26",
  "TraesCS2B03G0415300" = "#272e6a",
  "TraesCS2B03G0415600" = "#208a42",
  "TraesCS2B03G0415800" = "#89288f",
  "TraesCS2B03G0414600" = "#f47d2b",
  "TraesCS2B03G0415900" = "#fee500",
  "TraesCS2B03G0415100" = "#8a9fd1"
)

# ========== Custom plotting theme ==========

custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 12, face = "bold", color = "black", 
                               angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 14, face = "bold", color = "black"),
    legend.key.size = unit(0.8, "cm"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# ========== Plotting ==========

p <- ggplot(stat_data, aes(x = Treatment, y = mean, fill = Category)) +
  
  # 1. Draw bar chart
  geom_bar(
    stat = "identity",
    width = 0.6,
    position = position_dodge(width = 0.7),
    alpha = 0.8
  ) +
  
  # 2. Add error bars (standard error)
  geom_errorbar(
    aes(ymin = mean - se, ymax = mean + se, color = Category),
    width = 0.2,
    position = position_dodge(width = 0.7),
    size = 1,
    show.legend = FALSE
  ) +
  
  # 3. Add raw data points
  geom_point(
    data = data_long,
    aes(x = Treatment, y = Expression, color = Category),fen
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 2,
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  # 4. Color settings
  scale_fill_manual(values = gene_colors, name = "Gene") +
  scale_color_manual(values = gene_colors, name = "Gene") +
  
  # 5. Axis labels
  labs(
    x = "Treatment",
    y = "Expression Level"
  )

# Apply the plotting theme
b <- p + custom_theme_advanced

# Display plot
print(b)

# ========== Export the figure ==========

ggsave(
  filename = "Gene_Expression_Stress_Treatments.png",
  plot = b,
  width = 10,
  height =6,
  dpi = 300
)

ggsave(
  filename = "Gene_Expression_Stress_Treatments.pdf",
  plot = b,
  width = 14,
  height = 7
)

ggsave(
  filename = "Gene_Expression_Stress_Treatments.tiff",
  plot = b,
  width = 14,
  height = 7,
  dpi = 600,
  compression = "lzw"
)

#Fig.S5A
 
#gene haplotype visualization

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

setwd("gwas_result")
geno_target <- read.xlsx("TaUTP18_haplotype_raw_with_H_ddA276.xlsx")

# 1. automatically identify SNP column
snp_cols <- grep("^SNP\\.", names(geno_target), value = TRUE)

# 2. Ensure ID and SNP column is character type
geno_target <- geno_target %>%
  mutate(
    ID = as.character(ID),
    across(all_of(snp_cols), as.character)
  )

# 3. if missing Haplotype_raw, generate automatically
if (!"Haplotype_raw" %in% names(geno_target)) {
  geno_target <- geno_target %>%
    unite(
      col = "Haplotype_raw",
      all_of(snp_cols),
      sep = "_",
      remove = FALSE
    )
}

# 4. summarize each types of haplotype of frequency
hap_sum <- geno_target %>%
  group_by(Haplotype_raw, across(all_of(snp_cols))) %>%
  summarise(
    freq = n(),
    mean_H_vis_ddA = mean(H_vis_ddA.276, na.rm = TRUE),
    IDs = paste(ID, collapse = ", "),
    .groups = "drop"
  ) %>%
  arrange(desc(freq)) %>%
  mutate(Hap = paste0("Hap", row_number())) %>%
  relocate(Hap, .before = Haplotype_raw)

hap_sum
# TT -> T, CC -> C, TC -> T/C
format_genotype <- function(x) {
  x <- as.character(x)
  x_clean <- gsub("[^ACGT]", "", x)
  
  ifelse(
    nchar(x_clean) == 2 & substr(x_clean, 1, 1) == substr(x_clean, 2, 2),
    substr(x_clean, 1, 1),
    ifelse(
      nchar(x_clean) == 2,
      paste0(substr(x_clean, 1, 1), "/", substr(x_clean, 2, 2)),
      x
    )
  )
}

# 6. Organizein to long table used for ggplot
plot_df <- hap_sum %>%
  select(Hap, all_of(snp_cols), freq) %>%
  mutate(across(all_of(snp_cols), format_genotype)) %>%
  mutate(freq = as.character(freq)) %>%
  pivot_longer(
    cols = c(all_of(snp_cols), freq),
    names_to = "marker",
    values_to = "label"
  ) %>%
  mutate(
    marker = str_remove(marker, "^SNP\\."),
    marker = ifelse(marker == "freq", "freq", marker),
    fill_group = ifelse(marker == "freq", "freq", label),
    Hap = factor(Hap, levels = rev(hap_sum$Hap)),
    marker = factor(
      marker,
      levels = c(str_remove(snp_cols, "^SNP\\."), "freq")
    )
  )

# 7. Set colors
fill_colors <- c(
  "A" = "#F8766D",
  "T" = "#B07CFF",
  "C" = "#7CAE00",
  "G" = "#00BFC4",
  "A/T" = "#FDB462",
  "A/C" = "#FDB462",
  "A/G" = "#FDB462",
  "T/C" = "#FDB462",
  "T/G" = "#FDB462",
  "C/G" = "#FDB462",
  "freq" = "#E5E5E5"
)

# prevent new of genotype color missing
missing_cols <- setdiff(unique(plot_df$fill_group), names(fill_colors))
if (length(missing_cols) > 0) {
  fill_colors <- c(
    fill_colors,
    setNames(rep("#CCCCCC", length(missing_cols)), missing_cols)
  )
}

# 8. draw haplotype summary table
p <- ggplot(plot_df, aes(x = marker, y = Hap, fill = fill_group)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(
    aes(label = label),
    size = 5,
    color = "black",
    fontface = "bold"
  ) +
  scale_fill_manual(values = fill_colors, guide = "none") +
  labs(
    x = NULL,
    y = NULL,
    title = "Haplotype summary"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    
    # x axis SNP name
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      color = "black",
      face = "bold"
    ),
    
    # y axis Hap1 / Hap2 name
    axis.text.y = element_text(
      size = 13,
      color = "black",
      face = "bold"
    ),
    
    # title
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      color = "black"
    ),
    
    axis.title.x = element_text(
      color = "black",
      face = "bold"
    ),
    axis.title.y = element_text(
      color = "black",
      face = "bold"
    )
  ) +
  coord_fixed()
  
ggsave(
  filename = "haplotype_summary.png",
  plot = p,
  width = 3,
  height = 3,
  dpi = 300
)

#Fig.S5B

library(dplyr)
library(ggplot2)
library(ggpubr)
library(openxlsx)
library(scales)

setwd("gwas_result")

# ================= Import the data =================
plotData <- read.xlsx("TaUTP18_two_haplotype_plotData_corrected.xlsx")

# ================= Data Organize: rename haplotypes =================
plotData <- plotData %>%
  mutate(
    value = as.numeric(as.character(value)),
    

    value = ifelse(value == -9, NA, value),
    
    # original haplotype Export one column
    Haplotype_original = group,
    
    # haplotype rename
    group = case_when(
      group == "TT_CC_CC" ~ "Hap1",
      group == "CC_GG_AA" ~ "Hap2",
      TRUE ~ as.character(group)
    ),
    
    # Set order
    group = factor(group, levels = c("Hap1", "Hap2"))
  ) %>%
  filter(!is.na(value), !is.na(group))

# export renamed of Plotting Data
write.xlsx(plotData, "TaUTP18_two_haplotype_plotData_Hap1_Hap2.xlsx", rowNames = FALSE)


# ================= t-test test =================
t_res <- t.test(value ~ group, data = plotData)

cat("t-testresult:\n")
print(t_res)

t_table <- data.frame(
  group1 = "Hap1",
  group2 = "Hap2",
  n_Hap1 = sum(plotData$group == "Hap1"),
  n_Hap2 = sum(plotData$group == "Hap2"),
  mean_Hap1 = mean(plotData$value[plotData$group == "Hap1"], na.rm = TRUE),
  mean_Hap2 = mean(plotData$value[plotData$group == "Hap2"], na.rm = TRUE),
  median_Hap1 = median(plotData$value[plotData$group == "Hap1"], na.rm = TRUE),
  median_Hap2 = median(plotData$value[plotData$group == "Hap2"], na.rm = TRUE),
  t = as.numeric(t_res$statistic),
  df = as.numeric(t_res$parameter),
  p_value = t_res$p.value,
  conf_low = t_res$conf.int[1],
  conf_high = t_res$conf.int[2],
  method = t_res$method
)

write.xlsx(t_table, "TaUTP18_Hap1_Hap2_ttest_result.xlsx", rowNames = FALSE)

# ================= Calculate the y-axis range =================
y_min <- min(plotData$value, na.rm = TRUE)
y_max <- max(plotData$value, na.rm = TRUE)
y_range <- y_max - y_min

# prevent y_range as 0
if (y_range == 0) {
  y_range <- abs(y_max) * 0.1
}

signif_space <- y_range * 0.15
y_expand_lower <- 0.05
y_expand_upper <- 0.25

y_lower <- y_min - y_range * y_expand_lower
y_upper <- y_max + y_range * y_expand_upper

signif_y_position <- y_max + y_range * 0.12

cat("Y axis Set:\n")
cat("  Data range:", round(y_min, 6), "to", round(y_max, 6), "\n")
cat("  Display range:", round(y_lower, 6), "to", round(y_upper, 6), "\n")
cat("  Significance annotation position:", round(signif_y_position, 6), "\n")

# ================= Configure custom colors =================

custom_colors <- c(
  "Hap1" = "#1f77b4",
  "Hap2" = "#d62728"
)

# ================= Plotting section =================
temp_plot <- ggplot(plotData, aes(fill = group, y = value, x = group)) + 
  
  # Violin plot
  geom_violin(trim = TRUE, color = "gray", size = 0.8) +
  
  # Boxplot
  geom_boxplot(
    width = 0.2,
    position = position_dodge(0.9),
    size = 0.8
  ) +
  
  # Scatter points
  geom_point(
    position = position_jitterdodge(0.2),
    size = 2,
    alpha = 0.6
  ) +
  
  # Use custom colors
  scale_fill_manual(values = custom_colors) + 
  
  # Set Y axis limits
  scale_y_continuous(
    limits = c(y_lower, y_upper),
    expand = c(0, 0),
    breaks = scales::pretty_breaks(n = 6)
  ) +
  
  # Statistical test labels
  stat_compare_means(
    label = "p.signif",
    method = "t.test",
    size = 8,
    label.y = signif_y_position,
    label.x = 1.5,
    vjust = 0
  ) + 
  
  # X- and y-Axis labels
  xlab(expression(bold("TaUTP18-region haplotype"))) + 
  ylab(expression(bold("H_vis_ddA.276"))) +
  
  # Theme settings
  theme_bw(base_size = 18) +
  theme(
    text = element_text(color = "black", face = "bold", size = 18),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20, color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black", margin = margin(t = 10)),
    axis.title.y = element_text(size = 20, face = "bold", color = "black", margin = margin(r = 10)),
    axis.text.x = element_text(hjust = 0.5, vjust = 1, size = 16, color = "black", face = "bold"),
    axis.text.y = element_text(size = 16, color = "black", face = "bold"),
    axis.line = element_line(color = "black", size = 1),
    axis.ticks = element_line(color = "black", size = 0.8),
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(color = "black", size = 1.2, fill = NA),
    plot.margin = margin(t = 20, r = 10, b = 10, l = 10, unit = "pt")
  )


# ================= Export the figure =================

# PDF format
ggsave(
  "TaUTP18_Hap1_Hap2_H_ddA276_custom_colors.pdf",
  plot = temp_plot,
  width = 8,
  height = 7,
  dpi = 300
)

# PNG format
ggsave(
  "TaUTP18_Hap1_Hap2_H_ddA276_custom_colors.png",
  plot = temp_plot,
  width = 6,
  height = 5,
  dpi = 600
)

# TIFF format
ggsave(
  "TaUTP18_Hap1_Hap2_H_ddA276_custom_colors.tiff",
  plot = temp_plot,
  width = 8,
  height = 7,
  dpi = 600,
  compression = "lzw"
)

#Fig.S5C

LDBlockShow -InVCF gene_region_snp_vcf.vcf -Cutline 5 -InGWAS pvalue_H_vis_ddA276all -InGFF UTP18_2.gff3  -Region 5:151294896:151297502  -OutPut ./H_vis_ddA276_LDblock -OutPng  -SeleVar 2

#Fig.S5D
library(tidyverse)
library(ggrepel)
library(grid)
# ============================================================
# 1. Output directory
# ============================================================
out_dir <- "TaUTP18_2B_30variant_ATGup1.5kb_motif_structure"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# ============================================================
# 2. IWGSC Ref Seq v2.1 coordinates
# ============================================================

# Gene / TSS
gene_start <- 151296352
gene_end   <- 151297502

# 5'UTR
utr5_start <- 151296352
utr5_end   <- 151296395

# CDS
cds_start <- 151296396
cds_end   <- 151297502

# A of ATG = 0
ATG_A_pos <- cds_start

# ============================================================
# ATG upstream 1.5 kb
#
# ATG = 151296396
# ATG -1500 bp = 151294896
#
# Because 5'UTR = 44 bp:
#
# Upstream/TSS-upstream part:
# 151294896-151296351
# relative to ATG = -1500 to -45
#
# 5'UTR:
# 151296352-151296395
# relative to ATG = -44 to -1
# ============================================================

ATG_upstream_start <- ATG_A_pos - 1500

Upstream_start <- ATG_upstream_start
Upstream_end   <- gene_start - 1


PlantCARE_start <- gene_start - 2000
PlantCARE_end   <- gene_start - 1

# ============================================================
# 3. Gene structure
# A of ATG = 0
# ============================================================

gene_parts <- tibble(
  
  Region = c(
    "Upstream",
    "5'UTR",
    "CDS"
  ), 
  Genomic_Start = c(
    Upstream_start,
    utr5_start,
    cds_start
  ), 
  Genomic_End = c(
    Upstream_end,
    utr5_end,
    cds_end
  ) 
) %>%
  
  mutate(  
    Start = Genomic_Start - ATG_A_pos,
    End = Genomic_End - ATG_A_pos,    
    ymin = c(
      -0.07,
      -0.11,
      -0.16
    ),    
    ymax = c(
      0.07,
      0.11,
      0.16
    )
  )

# Expected:
#
# Upstream -1500 ~ -45
# 5'UTR -44 ~ -1
# CDS 0 ~ 1106


# ============================================================
# 4. 30 variants
#
# 16 upstream variants
# 14 CDS variants
#
# Total:
# 27 SNPs + 3 INDELs
#
# NOTE:
# SNP-12748049 and IND-00742842:
# true REF/ALT were not available in the previous
# real-allele table, so they are left as NA here.
# Genomic coordinates and motif overlaps remain unchanged.
# ============================================================

mut_raw <- tribble(
  
  ~Variant,         ~REF, ~ALT,          ~BP,        ~Region, 
  # ----------------------------------------------------------
  # ATG-upstream variants
  # ---------------------------------------------------------- 
  "SNP-12748048",   "T",  "C",           151294896,  "Upstream",  
  "SNP-12748049",   NA,   NA,            151294911,  "Upstream",  
  "SNP-12748050",   "T",  "C",           151294998,  "Upstream",  
  "SNP-12748051",   "T",  "C",           151295003,  "Upstream",  
  "SNP-12748052",   "A",  "G",           151295107,  "Upstream",  
  "SNP-12748053",   "C",  "T",           151295117,  "Upstream",  
  "SNP-12748054",   "G",  "A",           151295182,  "Upstream",  
  "SNP-12748055",   "C",  "G",           151295262,  "Upstream",  
  "SNP-12748057",   "G",  "C",           151295335,  "Upstream",  
  "IND-00742842",   NA,   NA,            151295435,  "Upstream",  
  "SNP-12748060",   "C",  "T",           151295564,  "Upstream",  
  "SNP-12748062",   "G",  "A",           151295681,  "Upstream",  
  "SNP-12748063",   "T",  "C",           151295823,  "Upstream",  
  "SNP-12748064",   "C",  "A",           151295951,  "Upstream",  
  "SNP-12748065",   "A",  "G",           151296068,  "Upstream",  
  "SNP-12748066",   "G",  "A",           151296173,  "Upstream",
 
  # ----------------------------------------------------------
  # CDS variants
  #
  # REF/ALT retained in data,
  # but will NOT be displayed in CDS labels.
  # ----------------------------------------------------------
  
  "SNP-12748069",   "G",  "C",           151296487,  "CDS",  
  "SNP-12748070",   "T",  "A",           151296511,  "CDS",  
  "SNP-12748071",   "T",  "C",           151296523,  "CDS",  
  "SNP-12748072",   "G",  "A",           151296542,  "CDS",  
  "SNP-12748073",   "A",  "G",           151296558,  "CDS",  
  "SNP-12748074",   "C",  "A",           151296570,  "CDS",  
  "SNP-12748076",   "A",  "C",           151296761,  "CDS",  
  "SNP-12748078",   "C",  "G",           151297217,  "CDS",  
  "SNP-12748079",   "T",  "G",           151297294,  "CDS",  
  "SNP-12748080",   "T",  "C",           151297316,  "CDS",  
  "IND-00742843",   "C",  "CG",          151297322,  "CDS",  
  "IND-00742844",   "G",  "GAGGACGACA",  151297326,  "CDS",  
  "SNP-12748081",   "T",  "C",           151297408,  "CDS",  
  "SNP-12748082",   "A",  "G",           151297455,  "CDS"
)


# ============================================================
# 5. Calculate variant coordinates
# ============================================================

mut_df <- mut_raw %>%  
  mutate(    
    # --------------------------------------------------------
    # Plot coordinate:
    # A of ATG = 0
    # -------------------------------------------------------- 
    Plot_position = BP - ATG_A_pos,      
    # --------------------------------------------------------
    # Relative to TSS
    # --------------------------------------------------------
    
    Relative_to_TSS = BP - gene_start,       
    # --------------------------------------------------------
    # Position in ATG-upstream 1.5 kb sequence
    # -------------------------------------------------------- 
    ATG_up1.5kb_position = if_else(
      
      Region == "Upstream",
      
      BP - ATG_upstream_start + 1,
      
      NA_real_
    ),
    
    # --------------------------------------------------------
    # Position in original Plant CARE 2-kb sequence
    #
    # IMPORTANT:
    # Plant CARE position 1 = Chr2B:151294352
    # -------------------------------------------------------- 
    PlantCARE_seq_position = if_else(
      
      Region == "Upstream",
      
      BP - PlantCARE_start + 1,
      
      NA_real_
    ),
    
    # --------------------------------------------------------
    # CDS position
    # A of ATG = CDS position 1
    # -------------------------------------------------------- 
    CDS_position = if_else(
      
      Region == "CDS",
      
      BP - cds_start + 1L,
      
      NA_integer_
    ),
       
    Mutation_detail = if_else(
      
      !is.na(REF) & !is.na(ALT),
      
      paste0(
        REF,
        "\u2192",
        ALT
      ),
      
      NA_character_
    ),
       
    # --------------------------------------------------------
    # Variant class
    # --------------------------------------------------------
    
    Variant_class = case_when(
      
      str_detect(Variant, "^IND-") ~ "INDEL",
      
      Region == "CDS" ~ "CDS SNP",
      
      Region == "Upstream" ~ "Upstream SNP",
      
      TRUE ~ "Variant"
    ),
    
    
    # --------------------------------------------------------
    # Labels
    #
    # CDS:
    # only show variant ID
    #
    # Upstream:
    # show ID + base substitution when REF/ALT known
    # --------------------------------------------------------
    
    Label = case_when(
      
      Region == "CDS" ~
        
        Variant,
      
      
      Region == "Upstream" &
        !is.na(Mutation_detail) ~
        
        paste0(
          Variant,
          "\n",
          Mutation_detail
        ),
      
      
      TRUE ~
        
        Variant
    )
  )


cat("\n30 variants:\n")

print(
  mut_df %>%
    select(
      Variant,
      Region,
      BP,
      Plot_position,
      Relative_to_TSS,
      ATG_up1.5kb_position,
      PlantCARE_seq_position,
      CDS_position,
      REF,
      ALT
    )
)


# ============================================================
# 6. Stagger variant heights
# ============================================================

Upstream_df <- mut_df %>%
  
  filter(
    Region == "Upstream"
  ) %>%
  
  arrange(
    Plot_position
  ) %>%
  
  mutate(
    
    Y_point = rep(
      
      c(
        0.38,
        0.53,
        0.68,
        0.83,
        0.98
      ),
      
      length.out = n()
    )
  )


cds_df <- mut_df %>%
  
  filter(
    Region == "CDS"
  ) %>%
  
  arrange(
    Plot_position
  ) %>%
  
  mutate(
    
    Y_point = rep(
      
      c(
        0.38,
        0.53,
        0.68,
        0.83,
        0.98
      ),
      
      length.out = n()
    )
  )


mut_df <- bind_rows(
  Upstream_df,
  cds_df
)


# ============================================================
# 7. Read Plant CARE motif file
#
# Plant CARE sequence:
# TSS upstream 2 kb
#
# sequence position 1 = Chr2B:151294352
# ============================================================

motif_file <-
  "plantCARE_output_PlantCARE_20289.csv"


motif_raw <- read.csv(
  
  motif_file,
  
  header = TRUE,
  
  check.names = FALSE,
  
  stringsAsFactors = FALSE,
  
  fileEncoding = "UTF-8"
)


cat("\nPlantCARE motif columns:\n")
print(colnames(motif_raw))


# If column names are not A-H:
if (!all(LETTERS[1:8] %in% colnames(motif_raw))) {
  
  colnames(motif_raw)[1:8] <- LETTERS[1:8]
}


# ============================================================
# Convert Plant CARE coordinates
# ============================================================

motif_df_all <- motif_raw %>%
  
  transmute(
    
    PlantCARE_ID = as.character(A),
    
    Family = as.character(B),
    
    Matched_sequence = as.character(C),
    
    Seq_start = as.numeric(D),
    
    Motif_Length = as.numeric(E),
    
    strand = as.character(F),
    
    Species = as.character(G),
    
    Annotation = as.character(H)
  ) %>%
  
  filter(
    
    !is.na(Seq_start),
    
    !is.na(Motif_Length)
  ) %>%
  
  mutate(
    
    Seq_end =
      Seq_start +
      Motif_Length -
      1,
    
    
    # --------------------------------------------------------
    # Convert Plant CARE 2-kb position to genomic coordinate
    #
    # position 1 = Plant CARE_start = 151294352
    # --------------------------------------------------------
    
    Genomic_Start =
      PlantCARE_start +
      Seq_start -
      1,
    
    Genomic_End =
      PlantCARE_start +
      Seq_end -
      1,
    
    
    # --------------------------------------------------------
    # Convert to ATG = 0 coordinate
    # --------------------------------------------------------
    
    Plot_Start =
      Genomic_Start -
      ATG_A_pos,
    
    Plot_End =
      Genomic_End -
      ATG_A_pos,
    
    Plot_Mid =
      (
        Plot_Start +
          Plot_End
      ) / 2,
    
    Motif_Y = 0,
    
    Label = paste0(
      Family,
      "\n",
      Seq_start,
      "-",
      Seq_end
    )
  ) %>%
  
  filter(
    
    Seq_start >= 1,
    
    Seq_end <= 2000
  )


# ============================================================
# Only retain motifs overlapping the displayed
# ATG-upstream region
#
# Displayed regulatory part:
# Chr2B:151294896-151296351
# ============================================================

motif_df <- motif_df_all %>%
  
  filter(
    
    Genomic_End >= Upstream_start,
    
    Genomic_Start <= Upstream_end
  )


cat("\nPlantCARE motifs retained in ATG-upstream display region:\n")

print(
  motif_df %>%
    count(
      Family,
      sort = TRUE
    )
)

write_csv(
  
  motif_df,
  
  file.path(
    out_dir,
    "TaUTP18_2B_ATGup1.5kb_PlantCARE_motifs.csv"
  )
)

# ============================================================
# 8. Automatically determine whether each variant
# is located inside a Plant CARE motif
#
# Avoid hard-coded SNP identifiers.
# ============================================================

mut_df <- mut_df %>%
  
  rowwise() %>%
  
  mutate(
    
    Motif_hit = {
      
      if (Region == "Upstream") {
        
        hits <- unique(
          
          motif_df$Family[
            
            BP >= motif_df$Genomic_Start &
              
              BP <= motif_df$Genomic_End
          ]
        )
        
        if (length(hits) == 0) {
          
          NA_character_
          
        } else {
          
          paste(
            hits,
            collapse = "; "
          )
        }
        
      } else {
        
        NA_character_
      }
    },
    
    In_motif =
      !is.na(Motif_hit)
  ) %>%
  
  ungroup()


cat("\nVariants located inside PlantCARE motifs:\n")

print(
  
  mut_df %>%
    
    filter(
      In_motif
    ) %>%
    
    select(
      Variant,
      BP,
      Plot_position,
      PlantCARE_seq_position,
      Motif_hit
    )
)


# Export overlap result
write_csv(
  
  mut_df,
  
  file.path(
    out_dir,
    "TaUTP18_2B_30variants_motif_overlap.csv"
  )
)


# ============================================================
# 9. Calculate distance between upstream variants and motifs
# ============================================================

snp_motif_distance <- mut_df %>%
  
  filter(
    Region == "Upstream"
  ) %>%
  
  select(
    Variant,
    BP,
    Plot_position,
    PlantCARE_seq_position
  ) %>%
  
  crossing(
    
    motif_df %>%
      
      select(
        Family,
        Seq_start,
        Seq_end,
        Plot_Start,
        Plot_End,
        Annotation
      )
  ) %>%
  
  mutate(
    
    Distance_to_motif = case_when(
      
      PlantCARE_seq_position >= Seq_start &
        PlantCARE_seq_position <= Seq_end ~ 0,
      
      PlantCARE_seq_position < Seq_start ~
        
        Seq_start -
        PlantCARE_seq_position,
      
      PlantCARE_seq_position > Seq_end ~
        
        PlantCARE_seq_position -
        Seq_end
    )
  ) %>%
  
  arrange(
    Variant,
    Distance_to_motif
  )


write_csv(
  
  snp_motif_distance,
  
  file.path(
    out_dir,
    "TaUTP18_2B_variant_to_PlantCARE_motif_distance.csv"
  )
)


cat("\nNearest motif for each upstream variant:\n")

print(
  
  snp_motif_distance %>%
    
    group_by(
      Variant
    ) %>%
    
    slice_min(
      Distance_to_motif,
      n = 1,
      with_ties = FALSE
    ) %>%
    
    ungroup()
)

# ============================================================
# 10. Colors
# ============================================================

region_colors <- c(
  
  "Upstream" = "grey85",
  
  "5'UTR" =
    "#66C2A5",
  
  "CDS" =
    "#4C78A8"
)


# ============================================================
# Plant CARE motif colors
# ============================================================

motif_families <-
  sort(
    unique(
      motif_df$Family
    )
  )


motif_colors <- setNames(
  
  grDevices::hcl.colors(
    
    n =
      length(
        motif_families
      ),
    
    palette =
      "Dark 3"
  ),
  
  motif_families
)


cat("\nMotif colors:\n")
print(motif_colors)


# ============================================================
# Variant shapes
# ============================================================

variant_shapes <- c(
  
  "Upstream SNP" = 16,
  
  "CDS SNP" = 18,
  
  "INDEL" = 15
)


# ============================================================
# 11. Plotting range
# ============================================================

plot_start <- -1530

plot_end <-
  cds_end -
  ATG_A_pos +
  30


# ============================================================
# 12. Main plot
# ============================================================

p <- ggplot() +
  
  
  # ----------------------------------------------------------
# Baseline
# ----------------------------------------------------------

annotate(
  
  "segment",
  
  x = plot_start,
  
  xend = plot_end,
  
  y = 0,
  
  yend = 0,
  
  linewidth = 0.8,
  
  color = "black"
) +
  
  
  # ----------------------------------------------------------
# Gene regions
# ----------------------------------------------------------

geom_rect(
  
  data = gene_parts,
  
  aes(
    
    xmin = Start,
    
    xmax = End,
    
    ymin = ymin,
    
    ymax = ymax,
    
    fill = Region
  ),
  
  color = "black",
  
  linewidth = 0.6
) +
  
  
  # ----------------------------------------------------------
# Plant CARE motifs
# ----------------------------------------------------------

geom_segment(
  
  data = motif_df,
  
  aes(
    
    x = Plot_Start,
    
    xend = Plot_End,
    
    y = 0,
    
    yend = 0,
    
    color = Family
  ),
  
  linewidth = 2.8,
  
  lineend = "butt",
  
  alpha = 0.90
) +
  
  
  # ----------------------------------------------------------
# CDS direction arrow
# ----------------------------------------------------------

annotate(
  
  "segment",
  
  x = 1106,
  
  xend = 1206,
  
  y = 0,
  
  yend = 0,
  
  arrow = arrow(
    
    length = unit(
      0.18,
      "cm"
    ),
    
    type = "closed"
  ),
  
  linewidth = 0.9,
  
  color = "black"
) +
  
  
  # ----------------------------------------------------------
# Upstream variant lollipop lines
# ----------------------------------------------------------

geom_segment(
  
  data =
    mut_df %>%
    filter(
      Region == "Upstream"
    ),
  
  aes(
    
    x = Plot_position,
    
    xend = Plot_position,
    
    y = 0.07,
    
    yend = Y_point
  ),
  
  color = "grey45",
  
  linewidth = 0.45
) +
  
  
  # ----------------------------------------------------------
# CDS variant lollipop lines
# ----------------------------------------------------------

geom_segment(
  
  data =
    mut_df %>%
    filter(
      Region == "CDS"
    ),
  
  aes(
    
    x = Plot_position,
    
    xend = Plot_position,
    
    y = 0.16,
    
    yend = Y_point
  ),
  
  color = "grey45",
  
  linewidth = 0.45
) +
  
  
  # ==========================================================
# VARIANT POINTS
#
# Black = not inside motif
# Red   = inside PlantCARE motif
# ==========================================================


# ----------------------------------------------------------
# Non-motif variants: BLACK
# ----------------------------------------------------------

geom_point(
  
  data =
    mut_df %>%
    filter(
      !In_motif
    ),
  
  aes(
    
    x = Plot_position,
    
    y = Y_point,
    
    shape = Variant_class
  ),
  
  color = "black",
  
  size = 3.5,
  
  alpha = 0.95
) +
  
  
  # ----------------------------------------------------------
# Motif-overlapping variants: RED
# ----------------------------------------------------------

geom_point(
  
  data =
    mut_df %>%
    filter(
      In_motif
    ),
  
  aes(
    
    x = Plot_position,
    
    y = Y_point,
    
    shape = Variant_class
  ),
  
  color = "#D73027",
  
  size = 4.2,
  
  alpha = 1
) +
  
  
  # ----------------------------------------------------------
# Upstream labels
#
# ID + mutation
#
# IMPORTANT:
# Restrict upstream labels to upstream region,
# so they will not enter the CDS region.
# ----------------------------------------------------------

geom_text_repel(
  
  data =
    mut_df %>%
    filter(
      Region == "Upstream"
    ),
  
  aes(
    
    x = Plot_position,
    
    y = Y_point,
    
    label = Label
  ),
  
  size = 3.5,
  
  fontface = "bold",
  
  color = "black",
  
  lineheight = 0.85,
  
  box.padding = 0.45,
  
  point.padding = 0.30,
  
  min.segment.length = 0,
  
  segment.color = "black",
  
  segment.linewidth = 0.65,
  
  force = 2.5,
  
  force_pull = 0.4,
  
  max.overlaps = Inf,
  
  
  # ========================================================
  # NEW:
  # upstream labels can only stay in the upstream region
  #
  # Upstream = -1500 to -45
  # ========================================================
  
  xlim = c(
    plot_start,
    -45
  ),
  
  
  seed = 123,
  
  show.legend = FALSE
) +
  
  
  
  # ----------------------------------------------------------
# CDS labels
#
# ONLY variant ID
# No REF/ALT
# No amino-acid annotation
# No CDS position
#
# IMPORTANT:
# Restrict CDS labels to CDS side.
# This prevents SNP-12748070 from covering
# upstream SNP-12748066.
# ----------------------------------------------------------

geom_label_repel(
  
  data =
    mut_df %>%
    filter(
      Region == "CDS"
    ),
  
  aes(
    
    x = Plot_position,
    
    y = Y_point,
    
    label = Label
  ),
  
  size = 3.3,
  
  fontface = "bold",
  
  color = "black",
  
  fill = "white",
  
  lineheight = 0.85,
  
  label.padding =
    unit(
      0.10,
      "lines"
    ),
  
  box.padding = 0.40,
  
  point.padding = 0.25,
  
  segment.color = "grey30",
  
  segment.linewidth = 0.60,
  
  force = 2.5,
  
  force_pull = 0.4,
  
  max.overlaps = Inf,
  
  
  # ========================================================
  # NEW:
  # CDS labels are restricted to x >= 0.
  #
  # Therefore SNP-12748070 cannot be pushed
  # into the upstream region and cover SNP-12748066.
  # ========================================================
  
  xlim = c(
    0,
    plot_end
  ),
  
  
  seed = 123,
  
  show.legend = FALSE
) +
  
  
  
  # ==========================================================
# Scales
# ==========================================================

scale_fill_manual(
  
  values = region_colors,
  
  name = "Gene region"
) +
  
  
  scale_color_manual(
    
    values = motif_colors,
    
    name = "Motif family"
  ) +
  
  
  scale_shape_manual(
    
    values = variant_shapes,
    
    name = "Variant type"
  ) +
  
  
  scale_x_continuous(
    
    breaks = c(      
      -1500,      
      -1000,     
      -500,      
      0,     
      500,      
      1000,      
      1106
    ),
    
    labels = c(
      
      "-1500",      
      "-1000",      
      "-500",     
      "0",      
      "+500",      
      "+1000",     
      "+1106"
    )
  ) +
  
  
  coord_cartesian(
    
    xlim =
      c(
        plot_start,
        plot_end
      ),
    
    ylim =
      c(
        -0.30,
        1.20
      ),
    
    clip = "off"
  ) +
  
  
# ==========================================================
# Labels
# ==========================================================

labs(
  
  x = NULL,
  
  y = NULL,
  
  caption =
    "Red points indicate variants overlapping PlantCARE-predicted motifs."
) +
  
  
# ==========================================================
# Theme
# ==========================================================

theme_classic(
  base_size = 16
) +
  
  theme(
    
    axis.title.x =
      element_blank(),
    
    axis.title.y =
      element_blank(),
    
    axis.text.x =
      element_text(
        
        size = 14,
        
        face = "bold",
        
        color = "black"
      ),
    
    axis.text.y =
      element_blank(),
    
    axis.ticks.y =
      element_blank(),
    
    axis.line.y =
      element_blank(),
    
    axis.line =
      element_line(
        
        color = "black",
        
        linewidth = 0.8
      ),
    
    axis.ticks =
      element_line(
        
        color = "black",
        
        linewidth = 0.8
      ),
    
    panel.grid.major.y =
      element_blank(),
    
    panel.grid.minor =
      element_blank(),
    
    legend.position =
      "bottom",
    
    legend.box =
      "vertical",
    
    legend.title =
      element_text(
        
        face = "bold",
        
        size = 13,
        
        color = "black"
      ),
    
    legend.text =
      element_text(
        
        size = 11,
        
        color = "black"
      ),
    
    plot.caption =
      element_text(
        
        size = 11,
        
        color = "black",
        
        hjust = 1
      ),
    
    plot.margin =
      margin(
        
        t = 5,        
        r = 20,        
        b = 5,        
        l = 15
      )
  ) +
  
  
  guides(
    
    fill =
      guide_legend(
        
        nrow = 1,
        
        byrow = TRUE
      ),
    
    color =
      guide_legend(
        
        nrow = 3,
        
        byrow = TRUE
      ),
    
    shape =
      guide_legend(
        
        nrow = 1,
        
        byrow = TRUE
      )
  )


# ============================================================
# 14. Export
# ============================================================

ggsave(
  
  file.path(
    out_dir,
    "TaUTP18_2B_30variant_ATGup1.5kb_PlantCARE_motif_map.png"
  ),
  
  plot = p,
  
  width = 18,
  
  height = 8,
  
  dpi = 800
)




#Fig.S5E

# ============================================================
# 9 SNPs haplotype plots for 2 traits
# Two big figures:
# 1) dd A276
# 2) Protein Content
#
# Each SNP is one panel.
# Within each SNP panel: C/H faceting with shared y-axis.
# Each treatment uses 2 homozygous haplotypes only.
# Colors: blue and red.
# X tick labels horizontal, no x-axis title.
# t-test p-value shown between 2 haplotypes in each treatment facet.
# ============================================================

library(tidyverse)
library(ggpubr)
library(stringr)
# ============================================================
# 1. path Set
# ============================================================
input_file <- "TaUTP18_30variants_ddA276_merged.tsv"
out_dir <- "haplotype_batch_plots_9SNP_2traits_bigfig"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# ============================================================
# 2. Import the data
# ============================================================
dat <- read.delim(
  input_file,
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  fileEncoding = "GB18030"
)


# ============================================================
# 3. specify 9 SNP
# ============================================================
snp_cols <- c(
  "SNP-12748053",
  "SNP-12748057",
  "SNP-12748052",
  "SNP-12748066",
  "SNP-12748050",
  "SNP-12748054",
  "SNP-12748062",
  "SNP-12748065",
  "IND-00742842"
)

missing_snps <- setdiff(snp_cols, colnames(dat))
if (length(missing_snps) > 0) {
  stop("below SNP column in Data cannot find to：\n", paste(missing_snps, collapse = ", "))
}

# ============================================================
# 4. two trait of phenotype Set
# ============================================================
trait_specs <- list(
  ddA276 = list(
    c_col = "C_vis_ddA.276",
    h_col = "H_vis_ddA.276",
    title = "ddA276",
    ylabel = "ddA276"
  ),
  Protein_Content = list(
    c_col = "C_Protein_Content",
    h_col = "H_Protein_Content",
    title = "Protein Content",
    ylabel = "Protein Content(%)"
  )
)


# ============================================================
# 5. single SNP × single trait Plotting Function
# ============================================================

make_snp_trait_plot <- function(dat, snp_col, pheno_c, pheno_h, trait_title) {
  
  # ==========================================================
  # 1. current trait phenotype long table
  # ==========================================================
  pheno_long <- dat %>%
    select(V1, all_of(c(pheno_c, pheno_h))) %>%
    pivot_longer(
      cols = all_of(c(pheno_c, pheno_h)),
      names_to = "TraitRaw",
      values_to = "value"
    ) %>%
    mutate(
      Treatment = if_else(str_starts(TraitRaw, "C_"), "C", "H"),
      Treatment = factor(Treatment, levels = c("C", "H")),
      value = suppressWarnings(as.numeric(value))
    ) %>%
    select(V1, Treatment, value)
  
  # ==========================================================
  # 2. current SNP / INDEL genotype
  # SNP: Keep only AA/CC/GG/TT
  # INDEL: Keep only actual homozygous type, and remove NA
  # ==========================================================
  
  geno_df <- data.frame(
    sample = dat$V1,
    genotype_raw = as.character(dat[[snp_col]]),
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      genotype_raw = str_trim(genotype_raw),
      genotype_raw = toupper(genotype_raw)
    )
  
  if (str_starts(snp_col, "IND")) {
    
    # INDEL process: 
    # 1) remove NA/empty value/missing
    # 2) Keep only“identical on both sides”of homozygous type
    # for example: 
    # G/G
    # GTCATGAATTTAGA/GTCATGAATTTAGA
    # but exclude: 
    # GTCATGAATTTAGA/G
    # <NA>
    geno_df <- geno_df %>%
      filter(
        !is.na(genotype_raw),
        genotype_raw != "",
        !genotype_raw %in% c("NA", ".", "./.", "0/0", "N/A")
      ) %>%
      separate(
        genotype_raw,
        into = c("allele1", "allele2"),
        sep = "/",
        remove = FALSE,
        fill = "right",
        extra = "merge"
      ) %>%
      filter(
        !is.na(allele1),
        !is.na(allele2),
        allele1 == allele2
      ) %>%
      transmute(
        sample = sample,
        genotype = genotype_raw
      )
    
  } else {
    
    # SNP process: 
    # Keep only AA/CC/GG/TT
    geno_df <- geno_df %>%
      mutate(
        genotype = str_replace_all(genotype_raw, "[/|\\s-]", "")
      ) %>%
      filter(
        genotype %in% c("AA", "CC", "GG", "TT")
      ) %>%
      transmute(
        sample = sample,
        genotype = genotype
      )
  }
  
  cat("\n", snp_col, "genotype after filtering:\n")
  print(table(geno_df$genotype, useNA = "ifany"))
  
  
  # ==========================================================
  # 3. combine genotype and phenotype
  # ==========================================================
  plot_df0 <- geno_df %>%
    inner_join(pheno_long, by = c("sample" = "V1")) %>%
    filter(
      !is.na(value),
      value != -9
    )
  
  if (nrow(plot_df0) == 0) {
    return(
      ggplot() +
        theme_void() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = paste0(snp_col, "\nNo valid homozygous data"),
          size = 5,
          fontface = "bold",
          color = "red"
        )
    )
  }
  
  count_df <- plot_df0 %>%
    group_by(Treatment, genotype) %>%
    summarise(
      num = n_distinct(sample),
      .groups = "drop"
    )
  
  cat("\n", snp_col, trait_title, "raw count by Treatment:\n")
  print(count_df)
  
  # Each Treatment must have at least 2 types of valid genotypes
  valid_treatments <- count_df %>%
    group_by(Treatment) %>%
    summarise(
      n_homo_types = n_distinct(genotype),
      .groups = "drop"
    ) %>%
    filter(n_homo_types >= 2) %>%
    pull(Treatment)
  
  if (length(valid_treatments) == 0) {
    return(
      ggplot() +
        theme_void() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = paste0(snp_col, "\nNo treatment has at least two valid haplotypes"),
          size = 4.5,
          fontface = "bold",
          color = "red"
        )
    )
  }
  
  count_df2 <- count_df %>%
    filter(Treatment %in% valid_treatments) %>%
    group_by(Treatment) %>%
    
    # if more than 2 types of valid genotypes, first take count  most of 2 types of
    arrange(desc(num), genotype, .by_group = TRUE) %>%
    slice_head(n = 2) %>%
    
    # finally sort by count from fewer to more
    arrange(num, genotype, .by_group = TRUE) %>%
    
    mutate(
      hap_order = row_number(),
      
      group_label = case_when(
        str_starts(snp_col, "IND") & genotype == "G/G" ~
          paste0("G/G(", num, ")"),
        
        str_starts(snp_col, "IND") & genotype != "G/G" ~
          paste0("INS/INS(", num, ")"),
        
        TRUE ~
          paste0(genotype, "(", num, ")")
      )
    ) %>%
    ungroup()
  
  
  
  cat("\n", snp_col, trait_title, "selected two haplotypes:\n")
  print(count_df2)
  
  # combine and Plotting Data
  plot_df <- plot_df0 %>%
    inner_join(
      count_df2 %>%
        select(Treatment, genotype, num, hap_order, group_label),
      by = c("Treatment", "genotype")
    ) %>%
    mutate(
      hap_order = factor(hap_order, levels = c(1, 2)),
      group_label = factor(group_label, levels = unique(count_df2$group_label))
    )
  
  # ==========================================================
  # 5. color: Each Treatment within No.1 types of homozygous type Blue, No.2 types of Red
  # ==========================================================
  hap_colors <- c(
    "1" = "#1F77B4",
    "2" = "#D62728"
  )
  
  # ==========================================================
  # 6. y axis limits: C/H share
  # ==========================================================
  y_min <- min(plot_df$value, na.rm = TRUE)
  y_max <- max(plot_df$value, na.rm = TRUE)
  y_range <- y_max - y_min
  
  if (y_range == 0 || !is.finite(y_range)) {
    y_range <- abs(y_max) * 0.1 + 1e-6
  }
  
  y_lower <- y_min - y_range * 0.05
  y_upper <- y_max + y_range * 0.32
  p_y_position <- y_max + y_range * 0.16
  
  # ==========================================================
  # 7. Each Treatment within eachly calculate Welch t-test p-value
  # ==========================================================
  pval_df <- plot_df %>%
    group_by(Treatment) %>%
    summarise(
      n_group = n_distinct(genotype),
      n1 = n_distinct(sample[hap_order == "1"]),
      n2 = n_distinct(sample[hap_order == "2"]),
      p_value = ifelse(
        n_group == 2 & n1 >= 3 & n2 >= 3,
        tryCatch(
          t.test(value ~ genotype)$p.value,
          error = function(e) NA_real_
        ),
        NA_real_
      ),
      .groups = "drop"
    ) %>%
    mutate(
      p_label = case_when(
        is.na(p_value) ~ "p = NA",
        p_value < 0.001 ~ paste0("p = ", formatC(p_value, format = "e", digits = 2)),
        TRUE ~ paste0("p = ", signif(p_value, 3))
      ),
      x_pos = 1.5,
      y_pos = p_y_position
    )
  
  # ==========================================================
  # 8. Plotting
  # ==========================================================
  p <- ggplot(
    plot_df,
    aes(
      x = group_label,
      y = value,
      fill = hap_order
    )
  ) +
    geom_violin(
      trim = TRUE,
      color = "gray40",
      linewidth = 0.7
    ) +
    geom_boxplot(
      width = 0.2,
      outlier.shape = NA,
      linewidth = 0.7,
      color = "black"
    ) +
    geom_point(
      position = position_jitter(width = 0.12, height = 0),
      size = 1,
      alpha = 0.60
    ) +
    
    geom_text(
      data = pval_df,
      aes(
        x = x_pos,
        y = y_pos,
        label = p_label
      ),
      inherit.aes = FALSE,
      size = 4.0,
      fontface = "bold",
      color = "black"
    ) +
    
    scale_fill_manual(values = hap_colors) +
    
    scale_y_continuous(
      limits = c(y_lower, y_upper),
      expand = c(0, 0),
      breaks = scales::pretty_breaks(n = 6)
    ) +
    
    facet_wrap(
      ~Treatment,
      nrow = 1,
      scales = "free_x"
    ) +
    
    labs(
      title = snp_col,
      x = NULL,
      y = NULL
    ) +
    
    theme_bw(base_size = 14) +
    theme(
      text = element_text(
        color = "black",
        face = "bold"
      ),
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 14,
        color = "black"
      ),
      
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      
      axis.text.x = element_text(
        hjust = 0.5,
        vjust = 0.5,
        size = 9,
        color = "black",
        face = "bold",
        angle = 0
      ),
      
      axis.text.y = element_text(
        size = 10,
        color = "black",
        face = "bold"
      ),
      
      strip.text = element_text(
        face = "bold",
        size = 11,
        color = "black"
      ),
      
      axis.line = element_line(
        color = "black",
        linewidth = 0.8
      ),
      
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.7
      ),
      
      legend.position = "none",
      
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      
      panel.border = element_rect(
        color = "black",
        linewidth = 0.9,
        fill = NA
      ),
      
      plot.margin = margin(
        t = 8,
        r = 6,
        b = 6,
        l = 6,
        unit = "pt"
      )
    )
  
  return(p)
}



# ============================================================
# 6. dd A276 and Protein Content
# ============================================================
for (trait_name in names(trait_specs)) {
  
  cat("\npositiveindraw trait:", trait_name, "\n")
  
  spec <- trait_specs[[trait_name]]
  
  plot_list <- lapply(snp_cols, function(snp) {
    make_snp_trait_plot(
      dat = dat,
      snp_col = snp,
      pheno_c = spec$c_col,
      pheno_h = spec$h_col,
      trait_title = spec$title
    )
  })
  
  # only for left most 3 subplot add y-axis title
  left_idx <- c(1, 4, 7)
  
  for (i in left_idx) {
    plot_list[[i]] <- plot_list[[i]] +
      labs(y = spec$ylabel) +
      theme(
        axis.title.y = element_text(
          size = 12,
          face = "bold",
          color = "black",
          angle = 90,
          margin = margin(r = 8)
        )
      )
  }
  
  
  # 9 SNP group combine 3x3 large plot
  p_big <- ggarrange(
    plotlist = plot_list,
    ncol = 3,
    nrow = 3,
    labels = NULL,
    common.legend = FALSE,
    align = "hv"
  )
  
  p_big <- annotate_figure(
    p_big,
    top = text_grob(
      paste0("Haplotype plots for ", spec$title),
      face = "bold",
      size = 18
    )
  )
  
  out_png <- file.path(out_dir, paste0(trait_name, "_16SNP_haplotype_bigplot.png"))
  out_pdf <- file.path(out_dir, paste0(trait_name, "_16SNP_haplotype_bigplot.pdf"))
  
  ggsave(
    out_png,
    plot = p_big,
    width = 12,
    height = 10,
    dpi = 400
  )
  
  ggsave(
    out_pdf,
    plot = p_big,
    width = 12,
    height = 10
  )


#Fig.S7A

#G26_2A expression level visualization

library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)

data <- read.xlsx("ddA276_for_plot.xlsx", 2)
data <- data[-1]
head(data)

# Change material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function 
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )
}
cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')")  
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +
        #scale_color_manual(values = custom_colors) +
        
        # ========== Statistical significance annotations ==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(3.5, 4.0, 4.5, 3.5, 4.0, 4.5),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 5))

# Display plot
print(b)

# Export the figure
ggsave("mutant_G26_2A_boxplot_4materials.png", b, width = 6, height = 5, dpi = 300)


#Fig.S7B
library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(openxlsx)
data <- read.xlsx("qRTPCR.xlsx", 4)
data <- data[-1]
head(data)

# Change material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function 
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')")  
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +
        #scale_color_manual(values = custom_colors) +
        
        # ========== Statistical significance annotations (Usereadyof variable)==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(2.5,3, 3.5,2.5, 3, 3.5),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 4))

# Display plot
print(b)

# Export the figure
ggsave("mutant_G26_2B_boxplot_4materials.png", b, width = 6, height = 5, dpi = 300)

#Fig.S7C

#G26_2D expression level visualization

library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(readxl)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 3)
data <- data[-1]
head(data)

# Change material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$Relative_expression <- as.numeric(data$Relative_expression)

# Compute summary statistics
stat_data <- data %>%
        group_by(Treatment, Material) %>%
        summarise( 
                n = n(),
                mean = mean(Relative_expression),
                sd = sd(Relative_expression),
                se = sd(Relative_expression)/sqrt(n),
                .groups = "drop"
        )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function
perform_ttests <- function(treatment_data, treatment_name) {
        bw_data <- treatment_data$Relative_expression[treatment_data$Material == "BW"]
        bb_data <- treatment_data$Relative_expression[treatment_data$Material == "bb"]
        dd_data <- treatment_data$Relative_expression[treatment_data$Material == "dd"]
        bbdd_data <- treatment_data$Relative_expression[treatment_data$Material == "bbdd"]
        
        if(length(bw_data) > 0 && length(bb_data) > 0) {
                ttest_bb <- t.test(bw_data, bb_data)
        } else {
                ttest_bb <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(dd_data) > 0) {
                ttest_dd <- t.test(bw_data, dd_data)
        } else {
                ttest_dd <- list(p.value = NA)
        }
        
        if(length(bw_data) > 0 && length(bbdd_data) > 0) {
                ttest_bbdd <- t.test(bw_data, bbdd_data)
        } else {
                ttest_bbdd <- list(p.value = NA)
        }
        
        return(data.frame(
                Treatment = treatment_name,
                Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
                p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
        ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
        mutate(
                significance = case_when(
                        is.na(p_value) ~ "ns",
                        p_value < 0.001 ~ "***",
                        p_value < 0.01 ~ "**",
                        p_value < 0.05 ~ "*",
                        TRUE ~ "ns"
                )
        )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
        if (sig == "***") {
                return("bold('***')")
        } else if (sig == "**") {
                return("bold('**')") 
        } else if (sig == "*") {
                return("bold('*')")
        } else {
                return("bold(ns)")
        }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
        theme(
                plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
                plot.tag = element_text(size = 14, face = "bold", color = "black"),
                axis.title.x = element_text(size = 20, face = "bold", color = "black"),
                axis.title.y = element_text(size = 20, face = "bold", color = "black"),
                axis.text.x = element_text(size = 18, face = "bold", color = "black"),
                axis.text.y = element_text(size = 18, face = "bold", color = "black"),
                axis.ticks.x = element_blank(),
                legend.title = element_text(size = 18, face = "bold", color = "black"),
                legend.text = element_text(size = 16, face = "bold", color = "black"),
                panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
                panel.grid.minor = element_blank(),
                panel.border = element_rect(color = "black", linewidth = 1)
        )

# Define 4 types of color
custom_colors <- c(
        "BW" = "#FF7F00",
        "bb" = "#004949",
        "dd" = "#009292",
        "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = Relative_expression, fill = Material)) +
        
        # Boxplot
        geom_boxplot(
                width = 0.6,
                position = position_dodge(width = 0.7),
                outlier.shape = NA,
                alpha = 0.8,
                size = 0.7
        ) +
        
        # Scatter points
        geom_point(
                color = "black",
                position = position_jitterdodge(
                        jitter.width = 0.1,
                        dodge.width = 0.7
                ),
                size = 1.5,
                alpha = 0.8,
                show.legend = FALSE
        ) +
        
        # Color settings
        scale_fill_manual(values = custom_colors) +        
        
        # ========== Statistical significance annotations ==========
geom_signif(
        xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
        xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
        annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                        annot_H_bb, annot_H_dd, annot_H_bbdd),
        y_position = c(1.5, 2.0, 2.5, 1.5, 2.0, 2.5),
        textsize = 7,
        vjust = 0.1,
        color = "black",
        tip_length = 0,
        parse = TRUE
) +
        
        labs(
                x = "Treatment",
                y = "Relative expression (2^-ΔΔCt)"
        )

# Apply the plotting theme
b <- p + 
        custom_theme_advanced + 
        expand_limits(y = c(0, 3))

# Display plot
print(b)

# Export the figure
ggsave("mutant_G26_2D_boxplot_4materials.png", b, width = 6, height = 5, dpi = 300)

#Fig.S8C

#mutant plant height visualization

library(ggplot2)
library(ggsci)
library(patchwork)
library(ggsignif)
library(dplyr)
library(openxlsx)


data <- read.xlsx("ddA276_for_plot.xlsx", 7)
data <- data[-1]
head(data)

# Change material order to BW, bb, dd, bbdd
data$Material <- factor(data$Material, levels = c("BW", "bb", "dd", "bbdd"))
data$PH <- as.numeric(data$PH)

# Compute summary statistics
stat_data <- data %>%
  group_by(Treatment, Material) %>%
  summarise( 
    n = n(),
    mean = mean(PH),
    sd = sd(PH),
    se = sd(PH)/sqrt(n),
    .groups = "drop"
  )

stat_data$Material <- factor(stat_data$Material, levels = c("BW", "bb", "dd", "bbdd"))

cat("Data summary:\n")
print(stat_data)

data$Group <- paste(data$Treatment, data$Material, sep = "_")
stat_data$Group <- paste(stat_data$Treatment, stat_data$Material, sep = "_")

# ttest function
perform_ttests <- function(treatment_data, treatment_name) {
  bw_data <- treatment_data$PH[treatment_data$Material == "BW"]
  bb_data <- treatment_data$PH[treatment_data$Material == "bb"]
  dd_data <- treatment_data$PH[treatment_data$Material == "dd"]
  bbdd_data <- treatment_data$PH[treatment_data$Material == "bbdd"]
  
  if(length(bw_data) > 0 && length(bb_data) > 0) {
    ttest_bb <- t.test(bw_data, bb_data)
  } else {
    ttest_bb <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(dd_data) > 0) {
    ttest_dd <- t.test(bw_data, dd_data)
  } else {
    ttest_dd <- list(p.value = NA)
  }
  
  if(length(bw_data) > 0 && length(bbdd_data) > 0) {
    ttest_bbdd <- t.test(bw_data, bbdd_data)
  } else {
    ttest_bbdd <- list(p.value = NA)
  }
  
  return(data.frame(
    Treatment = treatment_name,
    Comparison = c("bb_vs_BW", "dd_vs_BW", "bbdd_vs_BW"),
    p_value = c(ttest_bb$p.value, ttest_dd$p.value, ttest_bbdd$p.value)
  ))
}

treatment_c <- data[data$Treatment == "C", ]
treatment_h <- data[data$Treatment == "H", ]

ttest_results_c <- perform_ttests(treatment_c, "C")
ttest_results_h <- perform_ttests(treatment_h, "H")
ttest_results <- rbind(ttest_results_c, ttest_results_h)

# Add significance markers
ttest_results <- ttest_results %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "ns",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

cat("\n========== ttest results ==========\n")
print(ttest_results)

# ========== Prepare significance annotation text ==========
# Extract significance levels for each comparison
sig_C_bb <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bb_vs_BW"]
sig_C_dd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "dd_vs_BW"]
sig_C_bbdd <- ttest_results$significance[ttest_results$Treatment == "C" & ttest_results$Comparison == "bbdd_vs_BW"]
sig_H_bb <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bb_vs_BW"]
sig_H_dd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "dd_vs_BW"]
sig_H_bbdd <- ttest_results$significance[ttest_results$Treatment == "H" & ttest_results$Comparison == "bbdd_vs_BW"]

# Convert to parseable expressions
format_sig <- function(sig) {
  if (sig == "***") {
    return("bold('***')")
  } else if (sig == "**") {
    return("bold('**')")  
  } else if (sig == "*") {
    return("bold('*')")
  } else {
    return("bold(ns)")
  }
}

annot_C_bb <- format_sig(sig_C_bb)
annot_C_dd <- format_sig(sig_C_dd)
annot_C_bbdd <- format_sig(sig_C_bbdd)
annot_H_bb <- format_sig(sig_H_bb)
annot_H_dd <- format_sig(sig_H_dd)
annot_H_bbdd <- format_sig(sig_H_bbdd)

# Custom plotting theme
custom_theme_advanced <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 16, face = "bold", color = "black"),
    plot.tag = element_text(size = 14, face = "bold", color = "black"),
    axis.title.x = element_text(size = 20, face = "bold", color = "black"),
    axis.title.y = element_text(size = 20, face = "bold", color = "black"),
    axis.text.x = element_text(size = 18, face = "bold", color = "black"),
    axis.text.y = element_text(size = 18, face = "bold", color = "black"),
    axis.ticks.x = element_blank(),
    legend.title = element_text(size = 18, face = "bold", color = "black"),
    legend.text = element_text(size = 16, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "grey80", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1)
  )

# Define 4 types of color
custom_colors <- c(
  "BW" = "#FF7F00",
  "bb" = "#004949",
  "dd" = "#009292",
  "bbdd" = "#006ddb"
)

# ========== Plotting section ==========

p <- ggplot(data, aes(x = Treatment, y = PH, fill = Material)) +
  
  # Boxplot
  geom_boxplot(
    width = 0.6,
    position = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha = 0.8,
    size = 0.7
  ) +
  
  # Scatter points
  geom_point(
    color = "black",
    position = position_jitterdodge(
      jitter.width = 0.1,
      dodge.width = 0.7
    ),
    size = 1.5,
    alpha = 0.8,
    show.legend = FALSE
  ) +

  # Color settings
  scale_fill_manual(values = custom_colors) +
  
  # ========== Statistical significance annotations ==========
geom_signif(
  xmin = c(0.7, 0.7, 0.7, 1.7, 1.7,1.7),
  xmax = c(0.9, 1.1,1.3, 1.9, 2.1, 2.3),
  annotations = c(annot_C_bb, annot_C_dd, annot_C_bbdd, 
                  annot_H_bb, annot_H_dd, annot_H_bbdd),
  y_position = c(67, 68.5, 70, 61.5, 63, 64.5),
  textsize = 7,
  vjust = 0.1,
  color = "black",
  tip_length = 0,
  parse = TRUE
) +
  
  labs(
    x = "Treatment",
    y = "PH (cm)"
  )

# Apply the plotting theme
b <- p + 
  custom_theme_advanced + 
  expand_limits(y = c(50, 72))

# Display plot
print(b)

# Export the figure
ggsave("mutant_PH_boxplot_4materials.png", b, width = 10, height = 6, dpi = 300)
ggsave("mutant_PH_boxplot_4materials.pdf", b, width = 8, height = 7)
ggsave("mutant_PH_boxplot_4materials.tiff", b, width = 8, height = 7, dpi = 600, compression = "lzw")





#Fig.S12


library(tidyverse)
library(ggpubr)

#========================
# 1. Import the data
#========================
df <- read.csv(
  "merged_result.csv",
  fileEncoding = "GB18030",
  check.names = FALSE,
  stringsAsFactors = FALSE
)

#========================
# 2. select requires of column
#========================
df2 <- df %>%
  dplyr::select(
    `proteinwhite.B12_mean`,
    `C_Protein_Content(%)`
  )

#========================
# 3. Define Plotting Function
#========================
plot_cor_journal <- function(data, xvar, yvar, xlab, ylab, title_text) {
  p <- ggplot(data, aes(x = {{ xvar }}, y = {{ yvar }})) +
    geom_point(size = 3, color = "black", alpha = 0.85) +          # Black point
    geom_smooth(method = "lm", se = TRUE, 
                color = "red", fill = "#FDD0A2", linewidth = 1) + # Red line
    stat_cor(
      method = "pearson",
      label.x.npc = "left",
      label.y.npc = "top",
      size = 4.5,
      color = "black"
    ) +
    labs(
      x = xlab,
      y = ylab,
      title = title_text
    ) +
    theme_bw(base_size = 14) +   # slightly increase large basic font
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
      axis.text = element_text(color = "black", face = "bold", size = 12),   # Bold
      axis.title = element_text(color = "black", face = "bold", size = 14),  # Bold
      axis.line = element_line(color = "black", linewidth = 0.8),
      panel.grid = element_line(linetype = "dotted", color = "grey80")       
    )
  
  return(p)
}

#========================
# 4. draw P1 
#========================
p1 <- df2 %>%
  drop_na(`proteinwhite.B12_mean`, `C_Protein_Content(%)`) %>%
  plot_cor_journal(
    xvar = `proteinwhite.B12_mean`,
    yvar = `C_Protein_Content(%)`,
    xlab = "NIR-measured GPC (%)",                    
    ylab = "HSI-predicted GPC (%)",
    title_text = ""                          
  )


#========================
# 5. export high Resolution PNG
#========================
ggsave(
  filename = "population Near infrared/C_Protein_cor_plot1.png",
  plot = p1,
  width = 4,          
  height = 3,
  dpi = 300,
  bg = "white"
)



##Modeling
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import numpy as np

def CARS(X, y, iteration=300, n_comps=27 , cv=10):
    
    N, D = X.shape
    prob = 0.8
    a = np.power((D / 2), (1 / (iteration - 1)))
    k = (np.log(D / 2)) / (iteration - 1)
    r = [round(a * np.exp(-(k * i)) * D) for i in range(1, iteration + 1)]

    weights = np.ones(D) / D
    RMSECV = []
    idWs = []

    for i in range(iteration):
        idCal = np.random.choice(np.arange(N), size=int(prob * N), replace=False)
        idW = np.random.choice(np.arange(D), size=r[i], p=weights / weights.sum(), replace=False)
        idWs.append(idW)

        X_cal = X[idCal[:, np.newaxis], idW]
        Y_cal = y[idCal]
        comp = min(n_comps, len(idW))
        pls = PLSRegression(n_components=comp)
        pls.fit(X_cal, Y_cal)

        absolute = np.abs(pls.coef_).reshape(-1)
        weights[idW] = absolute / sum(absolute)
        MSE = -cross_val_score(pls, X_cal, Y_cal, cv=cv, scoring="neg_mean_squared_error")
        RMSE = np.mean(np.sqrt(MSE))
        RMSECV.append(RMSE)

    best_index = np.argmin(RMSECV)
    W_best = idWs[best_index]
    return W_best

def UVE(X, y, n_comps=27, cv=10):
    
    
    N, D = X.shape  # obtain sample number and number of variables
    # initialize PLS Model
    pls = PLSRegression(n_components=n_comps)
    # based on entire items Dataset fit combine PLS Model
    pls.fit(X, y)
    # Calculate Model of coefficient number
    original_coefs = np.abs(pls.coef_).reshape(-1)

    # through cross-validation Calculate Each variable of stability
    stability_scores = np.zeros(D)
    for i in range(D):
        # from variable set in order except current variable
        X_reduced = np.delete(X, i, axis=1)
        # refit combine PLS Model
        pls_cv = PLSRegression(n_components=n_comps)
        # Calculate cross-validation of MSE split number
        cv_scores = cross_val_score(pls_cv, X_reduced, y, cv=cv, scoring='neg_mean_squared_error')
        stability_scores[i] = np.mean(np.abs(cv_scores))

    # stability split number and original coefficients number of product determines variable of information property
    informative_scores = stability_scores * original_coefs
    # Apply the threshold to select informative predictor variables
    threshold = np.median(informative_scores)
    informative_vars = np.where(informative_scores >= threshold)[0]

    return informative_vars

def SPA(X, max_vars=250):
    #max_vars (maximum number of variables)as out put of features count
    # initialize parameter number
    n_samples, n_features = X.shape

    selected_vars = []
    remaining_vars = list(range(n_features))

    # random select No.one items features
    first_var = np.random.choice(remaining_vars)
    selected_vars.append(first_var)
    remaining_vars.remove(first_var)

    # Define positive orthogonal projection Function
    def orthogonal_projection(v, V_proj):
        if V_proj.shape[1] == 0:
            return np.zeros_like(v)
        projection = V_proj @ np.linalg.pinv(V_proj.T @ V_proj) @ V_proj.T @ v
        return projection

    # start iterations select other features
    while len(selected_vars) < max_vars and remaining_vars:
        max_residual = -np.inf
        next_var = None
        V_proj = X[:, selected_vars]

        # in remaining of features in Found minimum normalize projection residuals of features
        for var in remaining_vars:
            v = X[:, var]
            proj = orthogonal_projection(v, V_proj)
            residual = np.linalg.norm(v - proj)
            if residual > max_residual:
                max_residual = residual
                next_var = var

        if next_var is not None:
            selected_vars.append(next_var)
            remaining_vars.remove(next_var)
            V_proj = np.c_[V_proj, X[:, next_var]]  # update projection matrix

    return selected_vars


file_path = r"149A.xlsx" 
data = pd.read_excel(file_path,1)

X = data.iloc[:, 1:].values  # No.one column is label, remaining of is features
y = data.iloc[:, 0].values
print(X.shape)
print('programinrun，pleaseetc.pending···\n')



# apply CARS algorithm select features only Use one types of method！！！！
#selected_features_indices = CARS(X, y)
#selected_features_indices = UVE(X, y)
selected_features_indices = SPA(X)

# obtain original features name
original_feature_names = data.columns[1:]
# Use selected of features, and convertlabel column Add back to
X_selected = X[:, selected_features_indices]
selected_data_with_labels = np.column_stack((y, X_selected))
# obtain selected features of name
selected_feature_names = original_feature_names[selected_features_indices]
sorted_selected_features_indices = np.sort(selected_features_indices)
# Use sort after of selected features, and convert label column Add back to
X_selected_sorted = X[:, sorted_selected_features_indices]
selected_data_with_labels_sorted = np.column_stack((y, X_selected_sorted))
# obtain sort after selected features of name
sorted_selected_feature_names = original_feature_names[sorted_selected_features_indices]
# Export the selected features and labels to Excel, sorted by wavelength
selected_data_df_band_order = pd.DataFrame(selected_data_with_labels_sorted, columns=['target'] + list(sorted_selected_feature_names))
selected_data_df_band_order.to_excel('CARSoutputFile.xlsx', index=False)
print('process finished，filter result Save as：CARS output File.xlsx')


import pandas as pd
import numpy as np


class PPPPPretreatment:
    def msc(self, data):
        """
        positive ensure of MSC（multiple scatter correction positive）implement
        """
        if isinstance(data, np.ndarray) and data.dtype == np.object_:
            try:
                data = data.astype(float)
            except ValueError:
                print("Data in still save in unable to convert as float type of element，please check Data！")
                return data

        # Calculate average spectrum use as reference spectrum
        reference = np.mean(data, axis=0)

        new_data = []
        for sample in data:
            # for Each sample spectrum with reference spectrum perform linear regression
            
            coefficients = np.polyfit(reference, sample, 1)
            slope = coefficients[0]  # slope
            intercept = coefficients[1]  # intercept

            # MSC correction positive formula: correction positive after of spectrum = (original spectrum - intercept) / slope
            corrected_sample = (sample - intercept) / slope
            new_data.append(corrected_sample)

        return np.array(new_data)


# Read the Excel file
df = pd.read_excel(r"149A.xlsx",1)
# check Datastructure
print("Data shape:", df.shape)
print("first few Column names called:", df.columns[:5].tolist())

numeric_df = df.apply(pd.to_numeric)

# Extract reflectance data part, skip potentially of non-number value column
# Ensure only select number value type column
reflectivity_data = numeric_df.values
#reflectivity_data = df.iloc[:, 1:465].values

print("reflectance Data shape:", reflectivity_data.shape)
print("reflectance Data range - minimum value:", np.min(reflectivity_data), "maximumvalue:", np.max(reflectivity_data))

# instantiate preprocessing processclass
p = PPPPPretreatment()

# perform MSC preprocessing process
msc_processed = p.msc(reflectivity_data)

print("MSC process after Data range - minimum value:", np.min(msc_processed), "maximumvalue:", np.max(msc_processed))

# create new Data Frame; Export the results
result_df = pd.DataFrame(msc_processed, columns=df.columns)


# output to new Excel File
output_path = r"484_A_after_MSC.xlsx"
result_df.to_excel(output_path, index=False)

print("MSC process complete，Results saved to:", output_path)


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read the Excel file of different sheet
excel_file = r"149A+CARS.xlsx"
sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1')
sheet2 = pd.read_excel(excel_file, sheet_name='Sheet2')

# Ensure Column namesis a string
sheet1.columns = sheet1.columns.astype(str)
sheet2.columns = sheet2.columns.astype(str)

# Initialize a container for the results associated with each response variable
results_dict = {}
# initialize one items dictionary to store Each Y of best Model information
best_models_info = {}

# iterate over Sheet1 of each column (Assume these columns are potential Y)
for column in sheet1.columns:
    Y = sheet1[column]  # current column make as Y
    X = sheet2  # Sheet2 of Data make as X

    max_test_r2 = float('-inf')  # initialize Test set maximum R²as negative infinity, for subsequent comparisons and updates
    max_components = None  # initialize corresponding of number of principal components
    max_train_r2 = None  # Record the largest test set R² corresponding training set R²
    min_test_rmse = None  # Record the largest test set R²correspondingof Test set RMSE
    min_train_rmse = None  # Record the largest test set R² corresponding training set RMSE
    reference_std_y = Y.std()  # Calculate Y of standard deviation, used for subsequent RPD Calculate
    best_train_data = None  # Store the best test set R2 Corresponding training set Data (Contains X and Y)
    best_test_data = None  # Store the best test set R2 corresponding of Test set Data (Contains X and Y)
    best_pca_model = None  # used for store best PCA Model
    best_linear_model = None  # used for store best linear regression Model
    best_scaler = None  # used for store best normalization unit

    # iterate over 1 to 30 of number of principal components
    for n_components in range(1, 61):
        if n_components > X.shape[1]:
            continue

        # split Dataset (note: here we split again each time, but in practice it may use cross-validation)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=746)

        # create new normalizationunit
       # scaler = MinMaxScaler()

        # Step one: for self variable X perform normalization and principal components analysis (PCA)
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        # Step two: Use principal components after of result build linear regression Model
        model = LinearRegression()
        model.fit(X_train_pca, Y_train)

        # Generate predictions for the test set
        Y_pred_test = model.predict(X_test_pca)
        # Calculate test set of R^2value
        test_r2 = r2_score(Y_test, Y_pred_test)
        # Calculate test set of mean squared error (MSE)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        # Calculate test set of root mean squared error (RMSE), Use numpy of sqrt Function
        rmse_test = np.sqrt(mse_test)

        # Predict Training set
        Y_pred_train = model.predict(X_train_pca)
        # Compute the training-set R²
        train_r2 = r2_score(Y_train, Y_pred_train)
        # Calculate training set of mean squared error (MSE)
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        # Calculate training set of root mean squared error (RMSE), Use numpy of sqrt Function
        rmse_train = np.sqrt(mse_train)

        # if current Test set R² is greater than recorded of maximum Test set R², then update best metrics and corresponding number of principal components, also record Data
        if test_r2 > max_test_r2:
            max_test_r2 = test_r2
            max_components = n_components
            max_train_r2 = train_r2
            min_test_rmse = rmse_test
            min_train_rmse = rmse_train
            best_train_data = pd.concat(
                [pd.DataFrame(X_train, columns=X.columns), Y_train.reset_index(drop=True)], axis=1)
            best_test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), Y_test.reset_index(drop=True)],
                                       axis=1)
            best_pca_model = pca
            best_linear_model = model
            #best_scaler = scaler

    # Calculate test set RPD
    if min_test_rmse > 0:
        test_rpd = reference_std_y / min_test_rmse
    else:
        test_rpd = np.nan

    # Calculate training set RPD
    if min_train_rmse > 0:
        train_rpd = reference_std_y / min_train_rmse
    else:
        train_rpd = np.nan

    # take the current Y corresponding of each best metrics and number of principal components store in result dictionary
    results_dict[column] = {
        'Max_Test_R2_Score': max_test_r2,
        'N_Components': max_components,
        'Train_R2': max_train_r2,
        'Train_RMSE': min_train_rmse,
        'Test_RMSE': min_test_rmse,
        'Test_RPD': test_rpd,
        'Train_RPD': train_rpd
    }

    # store Model information
    best_models_info[column] = {
        'n_components': max_components,
        'pca_model': best_pca_model,
        'linear_model': best_linear_model,
        'scaler': best_scaler
    }

    # output current in specified order column of best metrics and corresponding of number of principal components
    print(f"\ncolumn {column} ofbestresult:")
    print(f"optimal number of principal components: {max_components}")
    print("Training set\t\t\tTest set")
    print(f"R2\tRMSE\tRPD\tR2\tRMSE\tRPD")
    print(
        f"{max_train_r2:.4f}\t{min_train_rmse:.4f}\t{train_rpd:.4f}\t{max_test_r2:.4f}\t{min_test_rmse:.4f}\t{test_rpd:.4f}")

    # convert best Test set R2 Corresponding training setand Test set Data Export to Excel File 
    with pd.ExcelWriter(f'Best_Data_{column}.xlsx') as writer:
        best_train_data.to_excel(writer, sheet_name='Train_Data', index=False)
        best_test_data.to_excel(writer, sheet_name='Test_Data', index=False)

# convert result convert as Data Frame and Export to Excel File
#results_df = pd.DataFrame.from_dict(results_dict, orient='index')
#results_file = 'PCR_Results.xlsx'
#results_df.to_excel(results_file)

#print(f"Results saved to {results_file}")

# Read the supplementary spectral dataset
extra_spectrum_file = r"A+CARS carbon content Predict.xlsx"
extra_spectrum_data = pd.read_excel(extra_spectrum_file, sheet_name='H')
extra_spectrum_data.columns = extra_spectrum_data.columns.astype(str)

# Use the trained model to predict additional spectral data Y
for y_name, model_info in best_models_info.items():
    n_components = model_info['n_components']
    pca_model = model_info['pca_model']
    linear_model = model_info['linear_model']
    scaler = model_info['scaler']

    # Use Export of normalization unit for additional outside Data perform normalization
    #extra_spectrum_scaled = scaler.transform(extra_spectrum_data)

    # Use Export of PCA Model for additional outside Data perform convert
    extra_spectrum_pca = pca_model.transform(extra_spectrum_data)

    # Use linear regression Model perform Predict
    extra_Y_pred = linear_model.predict(extra_spectrum_pca)

    # Export predictions to a new file
    result_df = pd.DataFrame({y_name: extra_Y_pred.flatten()})
    result_df.to_excel(f'Predicted_Y_{y_name}.xlsx', index=False)
    print(f"saved {y_name} of Predict result to Predicted_Y_{y_name}.xlsx")
	
	
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read different sheets from the original Excel file
excel_file = r"149A+UVE.xlsx"
sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1')
sheet2 = pd.read_excel(excel_file, sheet_name='Sheet2')

# Ensure Column names is a string
sheet1.columns = sheet1.columns.astype(str)
sheet2.columns = sheet2.columns.astype(str)

# used forstore Each Y of best Modelrelated results
best_models_info = {}

# iterate over Sheet1 of each column (Assume these columns are potential Y)
for column in sheet1.columns:
    Y = sheet1[column]
    X = sheet2

    max_test_r2 = float('-inf')
    best_n_components = None
    best_train_X = None
    best_train_Y = None

    for n_components in range(1, 61):
        if n_components > X.shape[1]:
            continue

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=736)
        pls = PLSRegression(n_components=n_components)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        pls.fit(X_train, Y_train)

        X_test = scaler.transform(X_test)
        Y_pred_test = pls.predict(X_test)
        test_r2 = r2_score(Y_test, Y_pred_test)

        if test_r2 > max_test_r2:
            max_test_r2 = test_r2
            best_n_components = n_components
            best_train_X = X_train.copy()
            best_train_Y = Y_train.copy()

    best_models_info[column] = {
        'n_components': best_n_components,
        'train_X': best_train_X,
        'train_Y': best_train_Y
    }

# Initialize an empty dictionary to store each Y of evaluation metric result
results_dict = {}

# again iterate over Sheet1 of each column, Calculate each evaluation metric
for column in sheet1.columns:
    Y = sheet1[column]
    X = sheet2

    max_test_r2 = float('-inf')
    max_components = None
    max_train_r2 = None
    min_test_rmse = None
    min_train_rmse = None
    reference_std_y = Y.std()

    for n_components in range(1, 61):
        if n_components > X.shape[1]:
            continue

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=736)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, Y_train)

        Y_pred_test = pls.predict(X_test)
        test_r2 = r2_score(Y_test, Y_pred_test)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        rmse_test = np.sqrt(mse_test)

        Y_pred_train = pls.predict(X_train)
        train_r2 = r2_score(Y_train, Y_pred_train)
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        rmse_train = np.sqrt(mse_train)

        if test_r2 > max_test_r2:
            max_test_r2 = test_r2
            max_components = n_components
            max_train_r2 = train_r2
            min_test_rmse = rmse_test
            min_train_rmse = rmse_train

    # Calculate test set RPD
    if min_test_rmse > 0:
        test_rpd = reference_std_y / min_test_rmse
    else:
        test_rpd = np.nan

    # Calculate training set RPD
    if min_train_rmse > 0:
        train_rpd = reference_std_y / min_train_rmse
    else:
        train_rpd = np.nan

    # take the current Y corresponding of each best metrics and number of principal components store in result dictionary
    results_dict[column] = {
        'Max_Test_R2_Score': max_test_r2,
        'N_Components': max_components,
        'Train_R2': max_train_r2,
        'Train_RMSE': min_train_rmse,
        'Test_RMSE': min_test_rmse,
        'Test_RPD': test_rpd,
        'Train_RPD': train_rpd
    }

   
# Read the supplementary spectral dataset
extra_spectrum_file = r"D:\Yu Jianming data\Prediction model construction and results\Predict\A+UVEproteincontentPredict.xlsx"
extra_spectrum_data = pd.read_excel(extra_spectrum_file,sheet_name='C')
extra_spectrum_data.columns = extra_spectrum_data.columns.astype(str)
extra_spectrum_data = scaler.transform(extra_spectrum_data)
#extra_spectrum_file = r"C protein content Predict.xlsx"
#extra_spectrum_data = pd.read_excel(extra_spectrum_file)
#extra_spectrum_data.columns = extra_spectrum_data.columns.astype(str)

# Use the trained model to predict additional spectral data Y
for y_name, model_info in best_models_info.items():
    n_components = model_info['n_components']
    train_X = model_info['train_X']
    train_Y = model_info['train_Y']

    pls = PLSRegression(n_components=n_components)
    pls.fit(train_X, train_Y)

    extra_Y_pred = pls.predict(extra_spectrum_data)

    # Export predictions to a new file
    result_df = pd.DataFrame({y_name: extra_Y_pred})
    result_df.to_excel(f'Predicted_Y_{y_name}.xlsx', index=False)
	
	
	
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Read the Excel file of different sheet
excel_file = r"149lgA+CARS.xlsx"
sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1')
sheet2 = pd.read_excel(excel_file, sheet_name='Sheet2')

# Ensure Column namesis a string
sheet1.columns = sheet1.columns.astype(str)
sheet2.columns = sheet2.columns.astype(str)

# Initialize a container for the results associated with each response variable
results_dict = {}
# initialize one items dictionary to store Each Y of best Model information
best_models_info = {}

# iterate over Sheet1 of each column (Assume these columns are potential Y)
for column in sheet1.columns:
    Y = sheet1[column]  # curren tcolumn make as Y
    X = sheet2  # Sheet2 of Data make as X

    max_test_r2 = float('-inf')  # initialize Test setmaximum R²as negative infinity, for subsequent comparisons and updates
    best_alpha = None  # initialize best of regularization parameter alpha
    max_train_r2 = None  # Record the best test set R² corresponding training set R²
    min_test_rmse = None  # Record the best test set R²correspondingof Test set RMSE
    min_train_rmse = None  # Record the best test set R² corresponding training set RMSE
    reference_std_y = Y.std()  # Calculate Y of standard deviation, used forsubsequent RPDCalculate
    best_train_data = None  # Store the best test set R² corresponding training set Data (Contains X and Y)
    best_test_data = None  # Store the best test set R²corresponding of Test set Data (Contains X and Y)
    best_model = None  # used for store best Model

    # iterate over one coefficient column regularization parameter alpha value 
    for alpha in [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]:

        # split Dataset (note: here we split again each time, but in practice it may use cross-validation)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=166)

        # create ridge back regression object, and Set current of regularization parameter alpha
        ridge_model = Ridge(alpha=alpha)

        # train Model
        ridge_model.fit(X_train, Y_train)

        # Generate predictions for the test set
        Y_pred_test = ridge_model.predict(X_test)
        # Calculate test set of R^2value
        test_r2 = r2_score(Y_test, Y_pred_test)
        # Calculate test set of mean squared error (MSE)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        # Calculate test set of root mean squared error (RMSE), Use numpy of sqrt Function
        rmse_test = np.sqrt(mse_test)

        # Predict Training set
        Y_pred_train = ridge_model.predict(X_train)
        # Compute the training-set R²
        train_r2 = r2_score(Y_train, Y_pred_train)
        # Calculate training set of mean squared error (MSE)
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        # Calculate training set of root mean squared error (RMSE), Use numpy of sqrt Function
        rmse_train = np.sqrt(mse_train)

        # if current Test set R² is greater than recorded of maximum Test set R², then update best metrics and corresponding regularization parameter alpha, also record Data
        if test_r2 > max_test_r2:
            max_test_r2 = test_r2
            best_alpha = alpha
            max_train_r2 = train_r2
            min_test_rmse = rmse_test
            min_train_rmse = rmse_train
            best_train_data = pd.concat([X_train, Y_train], axis=1)
            best_test_data = pd.concat([X_test, Y_test], axis=1)
            best_model = ridge_model  # Save best Model

    # Calculate test set RPD
    if min_test_rmse > 0:
        test_rpd = reference_std_y / min_test_rmse
    else:
        test_rpd = np.nan

    # Calculate training set RPD
    if min_train_rmse > 0:
        train_rpd = reference_std_y / min_train_rmse
    else:
        train_rpd = np.nan

    # take the current Y corresponding of each metric best metrics and regularization parameter alpha store in result dictionary
    results_dict[column] = {
        'Max_Test_R2_Score': max_test_r2,
        'Best_Alpha': best_alpha,
        'Train_R2': max_train_r2,
        'Train_RMSE': min_train_rmse,
        'Test_RMSE': min_test_rmse,
        'Test_RPD': test_rpd,
        'Train_RPD': train_rpd
    }

    # store Model information
    best_models_info[column] = {
        'model': best_model,
        'alpha': best_alpha
    }

    # output current in specified order column of best metrics and corresponding of regularization parameter alpha
    print(f"column {column} of")
    print("Training set\t\t\tTest set")
    print(f"R2\tRMSE\tRPD\tR2\tRMSE\tRPD")
    print(
        f"{max_train_r2:.4f}\t{min_train_rmse:.4f}\t{train_rpd:.4f}\t{max_test_r2:.4f}\t{min_test_rmse:.4f}\t{test_rpd:.4f}")

    # convert best Test set R² corresponding training set and Test set Data Export to Excel File (Each Y corresponding of Data Export as different work sheets)
    with pd.ExcelWriter(f'Best_Data_{column}.xlsx') as writer:
        best_train_data.to_excel(writer, sheet_name='Train_Data', index=False)
        best_test_data.to_excel(writer, sheet_name='Test_Data', index=False)

# convert result as Data Frameand Export to Excel File
results_df = pd.DataFrame.from_dict(results_dict, orient='index')
results_file = 'Ridge_Results.xlsx'
results_df.to_excel(results_file)

print(f"Results saved to {results_file}")
# Print the optimal alpha value for each response variable
print("\n each variable best regularization parameter summarize:")
for column, result in results_dict.items():
    print(f"{column}: best Alpha = {result['Best_Alpha']}")

# Read the supplementary spectral dataset
extra_spectrum_file = r"lgA+CARS protein white Predict_498.xlsx"
extra_spectrum_data = pd.read_excel(extra_spectrum_file, sheet_name='H')
extra_spectrum_data.columns = extra_spectrum_data.columns.astype(str)

# Use the trained model to predict additional spectral data Y
for y_name, model_info in best_models_info.items():
    model = model_info['model']

    # Use Export of Model perform Predict
    extra_Y_pred = model.predict(extra_spectrum_data)

    # Export predictions to a new file
    result_df = pd.DataFrame({y_name: extra_Y_pred.flatten()})
    result_df.to_excel(f'Predicted_Y_{y_name}.xlsx', index=False)
    print(f"saved {y_name} of Predict result to Predicted_Y_{y_name}.xlsx")
	
	
import numpy as np
import pandas as pd
from scipy import signal
import os


def moving_average_sg(data, window_size, order):
    # SG filtering
    sg_data = signal.savgol_filter(data, window_size, order)
    return sg_data


def process_csv_file(file_path, window_size, order, output_file):

    # extract all Column names
    column_names = df.columns.tolist()

    print(f"FileContains {len(column_names)} column")
    print(f"Column names: {column_names}")

    # check is no have enough of column
    if len(column_names) < 4:
        print("error：File column number less than 4 column，unable to remove No.2 column and No.3 column")
        return

    # remove No.2 column and No.3 column (index as 1 and 2)
    # retain No.one column and from No.four column start of all column
   # selected_columns = [column_names[0]] + column_names[3:]

    # Use No.one column make as x column Data, remaining column make as y column Data
    #x = df[column_names[0]].values
    #y_columns = selected_columns[1:]  # remove No.one column after of all column make as y column

    # create one items Data Frame used for Export all y column of smoothing result
    #df_result = pd.DataFrame({'x': x})
    smoothed_data = {}
    # Process each response-variable column
    for column in column_names:
        data = df[column].values

        # call wrapper of Savitzky-Golay filtering Function
        filtered_data = moving_average_sg(data, window_size, order)

        # convert smoothing result Add to Data Frame
        smoothed_data[f'{column}_SG'] = filtered_data
        df_result = pd.DataFrame(smoothed_data)

    # Export Data Frame to CSV File
    df_result.to_csv(output_file, index=False)
    print(f'Datasavedto {output_file}')


# Define Filepath
# Read different sheets from the original Excel file
df = pd.read_excel(r"149A.xlsx",1)

output_file = r'486_A_after_SG.csv'  # output CSVFile

# Process a single CSV file
process_csv_file(df, window_size=7, order=3, output_file=output_file)


import pandas as pd
import numpy as np


def snv_spectrum(spectra):
    """
    Calculate SNV for spectral data（standard normal variate transformation）

    Parameters:
    spectra: Two-dimensional array，each row represents one sample，each column represents one band

    Returns:
    snv_spectra: SNV-transformed spectral data
    """
    mean_value = np.mean(spectra, axis=1, keepdims=True)
    std_value = np.std(spectra, axis=1, keepdims=True)
    snv_spectra = (spectra - mean_value) / std_value
    return snv_spectra


# Assume the data frame is named df
# Read the Excel file
df = pd.read_excel(r"149A.xlsx",1)
#df = df.drop(df.columns[[1, 2]], axis=1)
# No.1 columnas Sample name, No.2-315 column is 1-314 bands of average reflectance

# Extract sample names
sample_names = df.iloc[:, 0]

# Extract reflectance data (No.2-315 column, corresponding 1-314 bands)
reflectance_data = df.iloc[:, 1:]

# Convert data to a Num Py array
reflectance_array = reflectance_data.values

# Calculate SNV values
snv_data = snv_spectrum(reflectance_array)

# Create a new data frame, Contains Sample name and SNV value
snv_df = pd.DataFrame(snv_data, columns=reflectance_data.columns)
snv_df.insert(0, df.columns[0], sample_names)  # Insert the sample-name column

# Export data to Excel
output_filename = r"SNV_processed_data.xlsx"
snv_df.to_excel(output_filename, index=False)

print(f"SNV processing completed; data exported to {output_filename}")
print(f"Data shape: {snv_df.shape}")
print(f"Number of samples: {len(sample_names)}")
print(f"Number of bands: {reflectance_data.shape[1]}")

